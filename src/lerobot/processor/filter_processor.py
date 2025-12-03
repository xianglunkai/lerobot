from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION

from .converters import from_tensor_to_numpy, to_tensor
from .core import EnvTransition, PolicyAction, RobotAction, TransitionKey
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

