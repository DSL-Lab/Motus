"""Offline adaptive cubic fitting, local parameter windows, and controller decoding.

Times are seconds on the original episode clock. All dimensions, including the
grippers, share knots. This module has no model, simulator, or Torch dependency.
"""

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline, make_lsq_spline


def normalize_actions(actions, minimum, maximum):
    span = np.asarray(maximum) - np.asarray(minimum)
    return np.where(span > 0, 2 * (actions - minimum) / np.where(span > 0, span, 1) - 1, 0)


def denormalize_actions(actions, minimum, maximum):
    return (actions + 1) * (np.asarray(maximum) - minimum) / 2 + minimum


def fit_adaptive_spline(times, actions, tolerance=0.02):
    """Insert knots at largest residuals until max elementwise error <= tolerance.

    A cubic interpolant is the exact fallback if greedy insertion exhausts the
    admissible data knots. This also handles sharp binary gripper transitions.
    """
    times, actions = np.asarray(times, float), np.asarray(actions, float)
    if (times.ndim != 1 or actions.ndim != 2 or len(times) != len(actions)
            or len(times) < 2 or not np.isfinite(times).all()
            or not np.isfinite(actions).all() or np.any(np.diff(times) <= 0)
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("Need finite actions, strictly increasing times, and positive tolerance")
    if len(times) < 4:
        grid = np.linspace(times[0], times[-1], 4)
        actions = np.stack([np.interp(grid, times, channel) for channel in actions.T], axis=1)
        times = grid
    interior = []
    candidates = times[2:-2]
    while True:
        knots = np.r_[np.repeat(times[0], 4), sorted(interior), np.repeat(times[-1], 4)]
        fitted = make_lsq_spline(times, actions, knots, k=3)
        residual = np.max(np.abs(fitted(times) - actions), axis=1)
        if residual.max() <= tolerance:
            return fitted
        available = candidates[~np.isin(candidates, interior)]
        if not len(available) or len(knots) - 4 >= len(times):
            return make_interp_spline(times, actions, k=3)
        worst_time = times[np.argmax(residual)]
        interior.append(available[np.argmin(np.abs(available - worst_time))])


def local_window(knots, controls, observation_time, rows, sample_dt):
    """Contiguous episode slice; last four control rows are structural padding.

    Extend beyond the clamped episode endpoint with a constant hold. Four equal
    endpoint knots separate the fitted trajectory from this constant extension;
    synthetic support knots increase by sample_dt rather than repeating forever.
    """
    if rows < 8 or sample_dt <= 0 or not knots[3] <= observation_time < knots[-4]:
        raise ValueError("Window needs >=8 rows and an observation before the episode end")
    span = min(np.searchsorted(knots, observation_time, side="right") - 1, len(controls) - 1)
    start = span - 3
    extended_knots = np.r_[knots, knots[-1] + sample_dt * np.arange(1, rows + 1)]
    extended_controls = np.concatenate([controls, np.repeat(controls[-1:], rows, axis=0)])
    return np.column_stack((extended_knots[start:start + rows] - observation_time,
                            extended_controls[start:start + rows]))


def select_video_frames(window, observation_time, frame_times, count, knot_stride):
    """Stride along future knot indices (3+s, 3+2s, ...); prepend current frame.

    Nearest timestamp wins; ties choose the earlier frame. Repeated rounded
    indices and requests beyond the episode are padding, even at the first tail
    request. No index ever crosses an episode boundary.
    """
    frame_times = np.asarray(frame_times, float)
    if (count < 1 or knot_stride < 1 or 3 + count * knot_stride > len(window) - 4
            or frame_times.ndim != 1 or len(frame_times) < 2
            or not np.isfinite(frame_times).all() or np.any(np.diff(frame_times) <= 0)):
        raise ValueError("Video selection must fit within the active spline domain")
    requested = np.r_[observation_time,
                      observation_time + window[3 + knot_stride * np.arange(1, count + 1), 0]]
    right = np.searchsorted(frame_times, requested).clip(0, len(frame_times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(np.abs(frame_times[left] - requested) <= np.abs(frame_times[right] - requested), left, right)
    valid = (requested >= frame_times[0]) & (requested <= frame_times[-1])
    valid[1:] &= indices[1:] > indices[:-1]
    return indices.astype(np.int64), valid


def repair_knots(times, controller_hz, max_duration):
    """Project unconstrained predictions to an anchored, ordered cubic domain.

    Keep the active left endpoint <=0 and first future knot >0. Strictly
    increasing suffix knots avoid excessive interior multiplicities after a
    running-maximum repair. The tiny spacing is 1e-5 controller ticks.
    """
    times = np.asarray(times, float).copy()
    if (times.ndim != 1 or len(times) < 8 or not np.isfinite(times).all()
            or not np.isfinite(controller_hz) or controller_hz <= 0
            or not np.isfinite(max_duration) or max_duration <= 0):
        raise ValueError("Invalid predicted knots or execution limits")
    gap = min(1e-5 / controller_hz, max_duration / (2 * len(times)))
    times = np.clip(times, -max_duration, max_duration - len(times) * gap)
    times[:4] = np.maximum.accumulate(np.minimum(times[:4], 0))
    for index in range(4, len(times)):
        times[index] = max(times[index], times[index - 1] + gap, gap)
    return times


def decode_window(parameters, metadata, execution_rows, *, repair=True):
    """Return commands, their relative timestamps, and next-call delay in seconds.

    H leading parameter rows support [u3,u(H-4)]. Execute ticks at 0 <= t <
    u(H-4); the half-open interval avoids executing the next plan's first tick
    twice. A positive sub-tick horizon yields one command. No time rescaling.
    """
    parameters = np.asarray(parameters, float)
    if (parameters.ndim != 2 or parameters.shape[1] != len(metadata["action_min"]) + 1
            or not np.isfinite(parameters).all()
            or not 8 <= execution_rows <= len(parameters)):
        raise ValueError("Invalid parameter shape or execution row budget")
    hz = float(metadata["controller_hz"])
    times = parameters[:, 0] * metadata["time_scale"]
    if repair:
        times = repair_knots(times, hz, metadata["max_duration"])
    horizon = min(float(times[execution_rows - 4]), metadata["max_duration"])
    if not times[3] <= 0 < horizon or np.any(np.diff(times) < 0):
        raise ValueError("Spline domain must contain the execution interval")
    count = max(1, int(np.ceil(horizon * hz - 1e-9)))
    ticks = np.arange(count) / hz
    spline = BSpline(times, parameters[:-4, 1:], 3, extrapolate=False)
    commands = spline(ticks)
    if not np.isfinite(commands).all():
        raise ValueError("Spline decoder produced nonfinite commands")
    commands = np.clip(commands, -1, 1)
    for index in metadata.get("gripper_indices", []):
        commands[:, index] = np.where(commands[:, index] >= 0, 1, -1)
    minimum = np.asarray(metadata["action_min"])
    commands = denormalize_actions(commands, minimum, np.asarray(metadata["action_max"]))
    return commands.astype(np.float32), ticks, count / hz
