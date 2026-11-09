"""Motus RoboTwin absolute-qpos adapter; scalar grippers never enter the spline."""

from time import perf_counter

import numpy as np

from .spline import FitError, evaluate_spline, fit_spline, reconstruction_error

ARM_CHANNELS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_CHANNELS = np.array([6, 13])


def _error_record(errors):
    # Failed numerical fits can contain NaN/Inf; keep strict JSON episode logs valid.
    return {
        key: [float(x) if np.isfinite(x) else None for x in values]
        for key, values in errors.items()
    }


def source_schedule(source_times, speed_factor, event_times=()):
    """Nominal execution grid mapped to source time, with exact endpoint/events.

    Source times include the observation anchor at t=0. The unchanged TOPP
    executor determines actual durations; these are waypoint sampling times.
    """
    t = np.asarray(source_times, dtype=float)
    if (
        t.ndim != 1
        or len(t) < 2
        or t[0] != 0
        or not np.isfinite(t).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("Expected increasing source timestamps starting at zero")
    if speed_factor not in (1.0, 1.5, 2.0):
        raise ValueError("speed_factor must be 1, 1.5, or 2")
    dt = t[1]
    if not np.allclose(np.diff(t), dt, rtol=1e-10, atol=1e-12):
        raise ValueError("Execution adapter requires uniformly sampled source commands")
    regular = (
        speed_factor * dt * np.arange(1, int(np.ceil(t[-1] / (speed_factor * dt))) + 1)
    )
    events = np.asarray(event_times, dtype=float)
    if not np.isfinite(events).all() or np.any(events <= 0) or np.any(events > t[-1]):
        raise ValueError("Gripper events must lie inside the executed source span")
    # Ignore near-end rounding, then append the original endpoint exactly once.
    regular = regular[regular < t[-1] - 1e-12 * max(1.0, t[-1])]
    schedule = np.unique(np.r_[regular, events, t[-1]])
    # Snap roundoff-near original samples so searchsorted selects the right event.
    for sample in t[1:]:
        schedule[np.isclose(schedule, sample, rtol=0, atol=1e-12)] = sample
    return np.unique(schedule)


def gripper_events(source_times, values, threshold=0.5):
    """Retain every open/closed edge, including repeated edges between grid points.

    Gripper values are raw scalars (0=closed, 1=open), held from the latest source
    sample. Threshold only identifies edges to insert; it never binarizes commands.
    """
    states = np.asarray(values) >= threshold
    edges = np.any(states[1:] != states[:-1], axis=1)
    return np.asarray(source_times)[1:][edges]


def decode_chunk(spline, query_times, gripper_times, gripper_values):
    """Reusable execution decoder for fitted OR directly predicted spline parameters."""
    gripper_times = np.asarray(gripper_times, dtype=float)
    gripper_values = np.asarray(gripper_values, dtype=float)
    if (
        gripper_times.ndim != 1
        or len(gripper_times) < 2
        or gripper_values.shape != (len(gripper_times), 2)
        or not np.isfinite(gripper_times).all()
        or not np.isfinite(gripper_values).all()
        or np.any(np.diff(gripper_times) <= 0)
    ):
        raise ValueError("Expected timestamped gripper samples [N,2]")
    commands = np.empty((len(query_times), 14), dtype=float)
    commands[:, ARM_CHANNELS] = evaluate_spline(spline, query_times)
    indices = np.searchsorted(gripper_times, query_times, side="right") - 1
    if np.any(indices < 0) or np.any(np.asarray(query_times) > gripper_times[-1]):
        raise ValueError("Gripper timeline does not cover requested timestamps")
    commands[:, GRIPPER_CHANNELS] = np.asarray(gripper_values)[indices]
    return commands


def adapt_chunk(
    actions,
    initial_state,
    *,
    action_adapter="dense",
    speed_factor=1.0,
    source_dt=0.1,
    tolerances=(0.01,) * 12,
    smoothing=1.0,
    max_control_points=None,
):
    """Return commands, optional spline, and diagnostics. No simulator dependencies.

    Full original chunk is the executed prefix. Dense 1x returns the exact input
    object, preserving values, dtype, order, and number of take_action calls.
    """
    if action_adapter not in ("dense", "spline") or speed_factor not in (1.0, 1.5, 2.0):
        raise ValueError("Unsupported adapter or speed")
    if not np.isfinite(source_dt) or source_dt <= 0:
        raise ValueError("source_dt must be positive")
    tol = np.asarray(tolerances, dtype=float)
    if tol.shape != (12,) or not np.isfinite(tol).all() or np.any(tol <= 0):
        raise ValueError("Expected twelve positive arm-joint tolerances in radians")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and nonnegative")
    if max_control_points is not None and (
        not isinstance(max_control_points, int) or max_control_points < 4
    ):
        raise ValueError("max_control_points must be an integer >=4")
    y = np.asarray(actions)
    initial = np.asarray(initial_state)
    if y.ndim != 2 or y.shape[1] != 14 or len(y) == 0 or initial.shape != (14,):
        raise ValueError("Expected actions [N,14] and initial state [14]")
    if not np.isfinite(y).all() or not np.isfinite(initial).all():
        raise ValueError("Nonfinite dense actions/state cannot be executed safely")
    t = np.arange(len(y) + 1, dtype=float) * source_dt
    samples = np.vstack([initial, y])
    grippers = samples[:, GRIPPER_CHANNELS]
    events = gripper_events(t, grippers)
    query = source_schedule(t, speed_factor, events)
    info = {
        "dense_samples": len(y),
        "fit_samples": len(samples),
        "source_span": float(t[-1]),
        "source_times": t.tolist(),
        "gripper_event_source_times": events.tolist(),
        "query_source_times": query.tolist(),
        "nominal_execution_times": (query / speed_factor).tolist(),
        "control_points": 0,
        "knots": 0,
        "fit_seconds": 0.0,
        "decode_seconds": 0.0,
        "fallback_count": 0,
        "errors": None,
        "failure": None,
    }
    if action_adapter == "dense" and speed_factor == 1:
        return actions, None, info
    start = perf_counter()
    dense = np.empty((len(query), 14), dtype=float)
    for channel in ARM_CHANNELS:
        dense[:, channel] = np.interp(query, t, samples[:, channel])
    dense[:, GRIPPER_CHANNELS] = grippers[np.searchsorted(t, query, side="right") - 1]
    info["decode_seconds"] = perf_counter() - start
    if action_adapter == "dense":
        return dense, None, info
    spline = None
    start = perf_counter()
    try:
        spline = fit_spline(
            t,
            samples[:, ARM_CHANNELS],
            tolerances,
            smoothing=smoothing,
            max_control_points=max_control_points,
        )
    except Exception as error:  # noqa: BLE001 -- numerical backend failures must use dense fallback
        info["failure"] = f"{type(error).__name__}: {error}"
        if isinstance(error, FitError) and error.errors is not None:
            info["errors"] = _error_record(error.errors)
    info["fit_seconds"] = perf_counter() - start
    if spline is not None:
        info.update(control_points=len(spline.control_points), knots=len(spline.knots))
        start = perf_counter()
        try:
            commands = decode_chunk(spline, query, t, grippers)
            errors = reconstruction_error(spline, t, samples[:, ARM_CHANNELS])
            # Also guard executed inter-sample queries against cubic overshoot.
            query_error = np.max(
                np.abs(commands[:, ARM_CHANNELS] - dense[:, ARM_CHANNELS]), axis=0
            )
            info["errors"] = _error_record({**errors, "query_max_abs": query_error})
            if not np.isfinite(commands).all() or np.any(
                query_error > np.asarray(tolerances)
            ):
                raise FitError("Execution-query reconstruction tolerance exceeded")
        except Exception as error:  # noqa: BLE001 -- invalid decoded fits must use dense fallback
            info["failure"] = f"{type(error).__name__}: {error}"
        info["decode_seconds"] += perf_counter() - start
        if info["failure"] is None:
            return commands, spline, info
    info["fallback_count"] = 1
    # At 1x, fallback is the exact original execution; above 1x, matched dense interpolation.
    return (actions if speed_factor == 1 else dense), spline, info
