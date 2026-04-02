from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
try:
    from filterpy.kalman import KalmanFilter as _ExternalKalman
    KalmanFilter = _ExternalKalman
except Exception:
    class KalmanFilter:
        """Lightweight fallback Kalman filter implementation.

        This minimal implementation provides `predict()` and `update(z)` and
        stores common matrices/attributes used by the processor (`x`, `P`,
        `F`, `H`, `Q`, `R`). It's intentionally small — use `filterpy` when
        available for advanced features and numerical robustness.
        """
        def __init__(self, dim_x: int, dim_z: int):
            self.dim_x = dim_x
            self.dim_z = dim_z
            self.F = np.eye(dim_x)
            self.H = np.zeros((dim_z, dim_x))
            self.Q = np.eye(dim_x)
            self.R = np.eye(dim_z)
            self.P = np.eye(dim_x)
            self.x = np.zeros((dim_x, 1))

        def predict(self) -> None:
            # x_k = F x_{k-1}
            self.x = self.F @ self.x
            # P_k = F P_{k-1} F^T + Q
            self.P = self.F @ self.P @ self.F.T + self.Q

        def update(self, z) -> None:
            z = np.array(z).reshape(self.dim_z, 1)
            # y = z - H x
            y = z - (self.H @ self.x)
            # S = H P H^T + R
            S = self.H @ self.P @ self.H.T + self.R
            # K = P H^T S^{-1}
            K = self.P @ self.H.T @ np.linalg.inv(S)
            # x = x + K y
            self.x = self.x + (K @ y)
            # P = (I - K H) P
            I = np.eye(self.dim_x)
            self.P = (I - K @ self.H) @ self.P
from torch import Tensor

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION

from .converters import from_tensor_to_numpy, to_tensor
from lerobot.types import EnvTransition, PolicyAction, RobotAction, TransitionKey
from .pipeline import PolicyProcessorPipeline, ProcessorStep, ProcessorStepRegistry

import math
import time


@dataclass
class _LowPassFilterMixin:
    """
    A mixin class providing core functionality for low-pass filtering.
    
    This class implements a first-order low-pass filter that can handle multi-dimensional
    tensors. The filter is applied independently to each dimension.
    
    Attributes:
        features: A dictionary mapping feature names to `PolicyFeature` objects.
        cutoff_freq: The cutoff frequency in Hz for the low-pass filter.
        dt: The time step in seconds between consecutive samples.
        device: The PyTorch device on which to store and perform tensor operations.
        dtype: The PyTorch dtype for tensor operations.
        eps: A small epsilon value for numerical stability.
        _last_value: Internal storage for the last filtered value.
        _last_time: Internal storage for the last processing time.
        _initialized: Flag indicating whether the filter has been initialized.
    """
    cutoff_freq: float = 1.0
    dt: float = 0.02
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    eps: float = 1e-8

    _last_value: dict[str, Tensor] = field(default_factory=dict, init=False, repr=False)
    _last_time: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _initialized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Initializes the mixin after dataclass construction."""
        if self.dtype is None:
            self.dtype = torch.float32
    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> _LowPassFilterMixin:
        """
        Moves the processor's state to the specified device and dtype.

        Args:
            device: The target PyTorch device.
            dtype: The target PyTorch dtype.

        Returns:
            The instance of the class, allowing for method chaining.
        """
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        
        # Move last_value tensors to the new device and dtype
        for key in self._last_value:
            self._last_value[key] = self._last_value[key].to(
                device=self.device, dtype=self.dtype
            )
        return self

    def state_dict(self) -> dict[str, Any]:
        """
        Returns the filter state as a flat state dictionary.

        Returns:
            A dictionary containing the filter state including last values and times.
        """
        state = {}
        for key, tensor in self._last_value.items():
            state[f"{key}.last_value"] = tensor.cpu()  # Always save to CPU
            state[f"{key}.last_time"] = self._last_time.get(key, 0.0)
        
        state["initialized"] = self._initialized
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Loads filter state from a state dictionary.

        Args:
            state: A state dictionary containing filter state.
        """
        self._last_value.clear()
        self._last_time.clear()
        
        for key, value in state.items():
            if key == "initialized":
                self._initialized = value
            elif key.endswith(".last_value"):
                feature_key = key.rsplit(".", 1)[0]
                self._last_value[feature_key] = value.to(device=self.device, dtype=self.dtype)
            elif key.endswith(".last_time"):
                feature_key = key.rsplit(".", 1)[0]
                self._last_time[feature_key] = value

    def get_config(self) -> dict[str, Any]:
        """
        Returns a serializable dictionary of the processor's configuration.

        Returns:
            A JSON-serializable dictionary containing the configuration.
        """
        return {
            "eps": self.eps,
            "cutoff_freq": self.cutoff_freq,
            "dt": self.dt,
        }

    def calc_lowpass_alpha_dt(self, dt: float, cutoff_freq: float) -> float:
        """
        Calculate the alpha parameter for the low-pass filter.

        Args:
            dt: Time step in seconds.
            cutoff_freq: Cutoff frequency in Hz.

        Returns:
            The alpha parameter for the filter.
        """
        if dt < 0.0 or cutoff_freq < 0.0:
            raise ValueError("dt and cutoff_freq must be non-negative")
        if cutoff_freq == 0.0:
            return 1.0  # No filtering
        if dt == 0.0:
            return 0.0  # Instant response
        
        rc = 1.0 / (2 * math.pi * cutoff_freq)
        return dt / (dt + rc)

    def reset(self, key: str, value: Tensor | None = None) -> None:
        """
        Reset the filter state for a specific key.

        Args:
            key: The feature key to reset.
            value: The value to reset to. If None, creates a zero tensor.
        """
        if value is None:
            # Create a zero tensor with the appropriate shape
            return
        
        self._last_value[key] = value.clone().detach()
        self._last_time[key] = time.perf_counter()
        self._initialized = True

    def _apply_lowpass_filter(
        self, 
        tensor: Tensor, 
        key: str, 
        feature_type: FeatureType,
        *,
        inverse: bool = False  # For API consistency, not used in filtering
    ) -> Tensor:
        """
        Apply low-pass filtering to a tensor.

        Args:
            tensor: The input tensor to filter.
            key: The feature key corresponding to the tensor.
            feature_type: The feature type (only ACTION is filtered).
            inverse: If True, bypasses filtering (for API consistency).

        Returns:
            The filtered tensor.
        """
        # Only filter actions, skip observations
        
        if feature_type != FeatureType.ACTION or inverse:
            return tensor

        # Initialize if needed
        if key not in self._last_value or not self._initialized:
            self.reset(key, tensor)

    
        # Calculate time difference
        current_time = time.perf_counter()
        dt = current_time - self._last_time.get(key, current_time)
        
        # Handle invalid time differences
        if dt <= 0.0 or dt > 1.0:  # Reset if dt is invalid or too large
            self.reset(key, tensor)
            return tensor

        # Calculate filter coefficient
        alpha = self.calc_lowpass_alpha_dt(dt, self.cutoff_freq)
        
        # Apply low-pass filter: y[n] = α * x[n] + (1-α) * y[n-1]
        filtered_value = alpha * tensor + (1.0 - alpha) * self._last_value[key]
        
        # Update state
        self._last_value[key] = filtered_value.clone().detach()
        self._last_time[key] = current_time

        return filtered_value

    def _filter_observation(self, observation: dict[str, Any]) -> dict[str, Tensor]:
        """
        Apply filtering to observation features (currently bypassed).

        Args:
            observation: The observation dictionary.

        Returns:
            The observation dictionary (unchanged, as we don't filter observations).
        """
        # Currently we don't filter observations, but the structure is here for future extension
        return observation

    def _filter_action(self, action: Tensor) -> Tensor:
        """
        Apply low-pass filtering to an action tensor.

        Args:
            action: The action tensor to filter.

        Returns:
            The filtered action tensor.
        """
        return self._apply_lowpass_filter(action, ACTION, FeatureType.ACTION)


@dataclass
@ProcessorStepRegistry.register(name="lowpass_filter_processor")
class LowPassFilterProcessor(_LowPassFilterMixin, ProcessorStep):
    """
    A processor step that applies low-pass filtering to actions in a transition.

    This class implements a first-order low-pass filter to smooth action outputs,
    which can help reduce high-frequency noise and create smoother robot motions.
    """

    @classmethod
    def from_lerobot_dataset(
        cls,
        dataset: LeRobotDataset,
        *,
        cutoff_freq: float = 1.0,
        dt: float = 0.02,
        eps: float = 1e-8,
        device: torch.device | str | None = None,
    ) -> LowPassFilterProcessor:
        """
        Creates a `LowPassFilterProcessor` instance.

        Args:
            dataset: The dataset (used for API consistency).
            features: The feature definition for the processor.
            cutoff_freq: The cutoff frequency in Hz for the low-pass filter.
            dt: The expected time step between samples in seconds.
            eps: A small epsilon value for numerical stability.
            device: The target device for the processor.

        Returns:
            A new instance of `LowPassFilterProcessor`.
        """
        return cls(
            cutoff_freq=cutoff_freq,
            dt=dt,
            eps=eps,
            device=device,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Apply low-pass filtering to the transition.

        Args:
            transition: The input environment transition.

        Returns:
            A new transition with filtered actions.
        """
        new_transition = transition.copy()

        # Handle action filtering
        action = new_transition.get(TransitionKey.ACTION)
        if action is not None:
            # translate to PolicyAction
            if isinstance(action, dict):
                robot_action = torch.tensor([action[key] for key in action.keys()])
            
            if not isinstance(robot_action, PolicyAction):
                raise ValueError(f"Action should be a PolicyAction type, got {type(robot_action)}")
            
            # Convert to tensor if needed and apply filtering
            robot_action_tensor = torch.as_tensor(robot_action, device=self.device, dtype=self.dtype)
            filtered_action = self._filter_action(robot_action_tensor)
            
            # back to RobotAction
            new_transition[TransitionKey.ACTION] = {name: filtered_action[i] for i, name in enumerate(action.keys())}

        return new_transition

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Transform the feature definitions (no transformation needed for filtering).

        Args:
            features: The input feature definitions.

        Returns:
            The same feature definitions (filtering doesn't change feature shapes/types).
        """
        return features


@dataclass
@ProcessorStepRegistry.register(name="kalman_filter_processor")
class KalmanFilterProcessor(ProcessorStep):
    """
    A processor step that applies a Kalman filter for state estimation.

    This processor estimates joint angles, velocities, and accelerations
    while accounting for noise in the system.
    """

    state_dim: int = 3  # [angle, velocity, acceleration]
    measurement_dim: int = 1  # Only angle is measured directly
    # process_noise: variance of the (continuous) acceleration/jerk noise
    process_noise: float = 1e-5
    # measurement_noise: variance of the angle measurement
    measurement_noise: float = 1e-2
    dt: float = 0.02  # Time step in seconds
    # Optional per-joint noise overrides
    per_joint_process_noise: dict[str, float] | None = None
    per_joint_measurement_noise: dict[str, float] | None = None
    # Enable simple adaptive R update using innovation squared moving average
    adaptive_R: bool = False
    adaptive_R_alpha: float = 0.1

    _kf: dict[str, KalmanFilter] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        """Initialize Kalman filters for each joint."""
        # Ensure _kf is a dictionary mapping joint names to KalmanFilter instances
        self._kf = {}

    def _initialize_kalman_filter(self, joint: str | None = None) -> KalmanFilter:
        """Create and initialize a Kalman filter for a single joint.

        If `joint` is provided, per-joint noise overrides will be applied.
        """
        kf = KalmanFilter(dim_x=self.state_dim, dim_z=self.measurement_dim)

        # State transition matrix (F)
        kf.F = np.array([
            [1, self.dt, 0.5 * self.dt**2],
            [0, 1, self.dt],
            [0, 0, 1]
        ])

        # Measurement function (H)
        kf.H = np.array([[1, 0, 0]])

        # Process noise covariance (Q)
        q_val = self.process_noise if (self.per_joint_process_noise is None or joint is None) else self.per_joint_process_noise.get(joint, self.process_noise)
        kf.Q = self._initialize_process_noise(q_val)

        # Measurement noise covariance (R)
        r_val = self.measurement_noise if (self.per_joint_measurement_noise is None or joint is None) else self.per_joint_measurement_noise.get(joint, self.measurement_noise)
        kf.R = self._initialize_measurement_noise(r_val)

        # Initial state covariance (P) -- set reasonable defaults
        kf.P = np.diag([max(self.measurement_noise, 1e-6), 1.0, 1.0])

        # Initial state (x)
        kf.x = np.zeros((self.state_dim, 1))

        return kf

    def _initialize_process_noise(self, q: float) -> np.ndarray:
        """Dynamically initialize the process noise covariance matrix (Q).

        `q` is the variance of the acceleration white noise.
        """
        q = float(q)
        dt = float(self.dt)
        q11 = (dt ** 4) / 4.0
        q12 = (dt ** 3) / 2.0
        q13 = (dt ** 2) / 2.0
        q22 = dt ** 2
        q23 = dt
        q33 = 1.0

        Q = q * np.array([
            [q11, q12, q13],
            [q12, q22, q23],
            [q13, q23, q33],
        ], dtype=np.float64)
        return Q

    def _initialize_measurement_noise(self, r: float) -> np.ndarray:
        """Dynamically initialize the measurement noise covariance matrix (R).

        `r` is the measurement variance for the angle.
        """
        return np.array([[float(r)]], dtype=np.float64)

    def reset(self, joint_names: list[str], initial_states: dict[str, np.ndarray] | None = None):
        """Reset Kalman filters for all joints."""
        self._kf.clear()
        for joint in joint_names:
            self._kf[joint] = self._initialize_kalman_filter()
            initial_state = initial_states.get(joint) if initial_states else None
            self._kf[joint].x = initial_state if initial_state is not None else np.zeros((self.state_dim, 1))

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        """
        Apply Kalman filters to estimate the state for multiple joints.

        Args:
            transition: The input environment transition.

        Returns:
            A new transition with estimated states for all joints.
        """
        new_transition = transition.copy()

        # Extract the measurements (e.g., joint angles).
        # Support both enum-keyed and string-keyed transitions.
        measurements = new_transition.get(TransitionKey.OBSERVATION)
        if measurements is None:
            measurements = new_transition.get(TransitionKey.OBSERVATION.value)
        if measurements is None:
            raise ValueError("Measurements are missing in the transition.")

        # Ensure measurements are a dictionary of joint names to values
        if not isinstance(measurements, dict):
            raise ValueError("Measurements should be a dictionary of joint names to values.")

        estimated_states = {}
        for joint, measurement in measurements.items():
            # Ensure measurement is a numpy array
            measurement_np = np.array(measurement, dtype=np.float32).reshape(-1, 1)

            # Initialize per-joint filter and align its initial state with first measurement
            if joint not in self._kf:
                kf = self._initialize_kalman_filter(joint)
                # Align angle state to the first measurement to avoid bad initial prediction
                angle_init = float(measurement_np[0, 0])
                kf.x = np.array([[angle_init], [0.0], [0.0]], dtype=np.float32)
                # Set initial covariance: low uncertainty on measured angle, larger on vel/acc
                # Use the filter's R (measurement variance) to initialize angle variance
                angle_var = float(kf.R[0, 0]) if hasattr(kf, "R") else float(max(self.measurement_noise, 1e-5))
                vel_var = 1.0
                acc_var = 1.0
                kf.P = np.diag([angle_var, vel_var, acc_var]) * 10.0
                self._kf[joint] = kf

            # Kalman filter predict and update steps
            self._kf[joint].predict()
            # compute innovation and optionally update R adaptively after update
            # call update and capture innovation if supported by external filter; our lightweight
            # fallback does not return innovation, so compute predicted measurement and residual here.
            # predicted measurement: H x
            pred_z = (self._kf[joint].H @ self._kf[joint].x).reshape(-1, 1)
            residual = measurement_np - pred_z
            self._kf[joint].update(measurement_np)

            # Adaptive R: simple exponential moving average of innovation^2 + HpH^T
            if self.adaptive_R:
                innov_sq = float((residual ** 2).mean())
                HpHT = float(self._kf[joint].H @ self._kf[joint].P @ self._kf[joint].H.T)
                measured_innov_var = innov_sq + HpHT
                old_R = float(self._kf[joint].R[0, 0])
                new_R = (1.0 - self.adaptive_R_alpha) * old_R + self.adaptive_R_alpha * measured_innov_var
                self._kf[joint].R = np.array([[new_R]], dtype=np.float64)

            # Store the estimated state
            estimated_state = self._kf[joint].x.flatten()
            estimated_states[joint] = {
                "angle": float(estimated_state[0]),
                "velocity": float(estimated_state[1]),
                "acceleration": float(estimated_state[2])
            }

        # Update the transition with the estimated states
        # `TransitionKey` does not define a STATE key; store under COMPLEMENTARY_DATA
        comp_key = TransitionKey.COMPLEMENTARY_DATA.value
        comp = deepcopy(new_transition.get(comp_key) or {})
        comp["kalman_states"] = estimated_states
        new_transition[comp_key] = comp

        return new_transition

    def get_config(self) -> dict[str, Any]:
        """Return the configuration of the Kalman filter."""
        return {
            "state_dim": self.state_dim,
            "measurement_dim": self.measurement_dim,
            "process_noise": self.process_noise,
            "measurement_noise": self.measurement_noise,
            "dt": self.dt
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """
        Minimal implementation to satisfy the abstract `ProcessorStep` interface.

        For the Kalman filter processor we don't change feature shapes/types,
        so return the input features unchanged.
        """
        return features

