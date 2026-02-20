import numpy as np

from lerobot.processor.filter_processor import KalmanFilterProcessor
from lerobot.processor.core import TransitionKey


def test_kalman_filter_multi_joint_simulation():
    """Simulate 7 joints over 10 seconds and verify KF reduces RMSE.

    The test creates smooth sinusoidal true signals for 7 joints, adds
    Gaussian measurement noise, feeds noisy measurements to
    `KalmanFilterProcessor`, and asserts the filtered estimates have lower
    RMSE relative to the true signals than the raw noisy measurements.
    """
    rng = np.random.RandomState(0)

    dt = 0.02
    duration = 10.0
    steps = int(duration / dt)
    t = np.arange(steps) * dt

    n_joints = 7
    joint_names = [f"joint_{i}" for i in range(n_joints)]

    true_signals = {}
    measurements = {}

    # Build per-joint true signals and noisy measurements
    for i, name in enumerate(joint_names):
        amp = 0.5 + 0.1 * i
        freq = 0.1 + 0.02 * i
        phase = i * 0.2
        true = amp * np.sin(2 * np.pi * freq * t + phase)
        noise_std = 0.05
        meas = true + rng.normal(scale=noise_std, size=steps)
        true_signals[name] = true
        measurements[name] = meas

    # Instantiate processor (measurement_noise expects variance)
    proc = KalmanFilterProcessor(process_noise=1e-5, measurement_noise=noise_std ** 2, dt=dt)

    # Run simulation: feed measurements step-by-step
    estimates = {name: np.zeros(steps) for name in joint_names}

    for step in range(steps):
        meas_dict = {name: float(measurements[name][step]) for name in joint_names}
        transition = {TransitionKey.OBSERVATION: meas_dict}
        out = proc(transition)
        comp = out.get(TransitionKey.COMPLEMENTARY_DATA.value, {})
        states = comp.get("kalman_states", {})
        for name in joint_names:
            estimates[name][step] = states[name]["angle"]

    # Compute RMSE for raw measurements and for Kalman estimates (average across joints)
    rmse_raw_per_joint = [np.sqrt(np.mean((measurements[n] - true_signals[n]) ** 2)) for n in joint_names]
    rmse_kf_per_joint = [np.sqrt(np.mean((estimates[n] - true_signals[n]) ** 2)) for n in joint_names]

    rmse_raw = float(np.mean(rmse_raw_per_joint))
    rmse_kf = float(np.mean(rmse_kf_per_joint))

    print(f"RMSE raw: {rmse_raw:.6f}, RMSE KF: {rmse_kf:.6f}")

    # Expect Kalman filter to improve (reduce) RMSE in this simple test
    assert rmse_kf < rmse_raw
