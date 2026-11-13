"""Episode-level cubic fitting; no model, simulator, or image dependencies.

Insertion follows bspline-policy/common/bspline_action.py: SciPy generate_knots
(FITPACK residual-region insertion), followed by explicit-knot QR least squares.
Unlike the reference wrapper, exhausted/capped fits remain failed fits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import BSpline, generate_knots, make_lsq_spline

VERSION = "robotwin-episode-cubic-v1"
CHANNELS = [f"left_joint_{i}" for i in range(1, 7)] + ["left_gripper"] + [
    f"right_joint_{i}" for i in range(1, 7)
] + ["right_gripper"]


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_actions(path):
    """Read precisely the original MOTUS target, without touching images."""
    path = Path(path).resolve(strict=True)
    if path.suffix in (".hdf5", ".h5"):
        import h5py
        with h5py.File(path, "r") as f:
            actions = f["joint_action/vector"][()]
    elif path.suffix == ".pt":
        import torch
        actions = torch.load(path, map_location="cpu", weights_only=True).numpy()
    elif path.suffix == ".npy":
        # Explicit action-only interchange / synthetic test input.
        actions = np.load(path, allow_pickle=False)
    else:
        raise ValueError(f"Unsupported episode format: {path}")
    if path.suffix != ".npy":
        # Original raw converter and training loader both cast commands to float32.
        actions = np.asarray(actions, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 4:
        raise ValueError(f"Expected finite (T>=4,14) commanded actions: {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Nonfinite source actions")
    return actions


def source_identity(path, actions):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                action_sha256=hashlib.sha256(actions.tobytes()).hexdigest())


def normalize(actions, norm):
    low, high = np.array(norm["low"]), np.array(norm["high"])
    width = high - low
    y = 2 * (actions - low) / np.where(width == 0, 1, width) - 1
    # A training-constant channel cannot encode an outlier by mapping it to zero.
    if np.any(actions[:, width == 0] != low[width == 0]):
        raise ValueError("Episode differs from a training-constant channel; normalization cannot represent it")
    y[:, width == 0] = 0
    return y


def decode(y, norm):
    low, high = np.array(norm["low"]), np.array(norm["high"])
    return (y + 1) * (high - low) / 2 + low


def fit_episode(y, tolerance=0.01, max_control_points=None, smoothing=1e-12):
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 2 or len(y) < 4 or not np.isfinite(y).all():
        raise ValueError("Need at least four finite multichannel samples")
    if not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("Tolerance and generator smoothing must be finite and positive")
    cap = min(len(y), max_control_points if max_control_points is not None else len(y))
    if cap < 4:
        raise ValueError("Control point cap must be >=4")
    x = np.arange(len(y), dtype=float)
    history, spline = [], None
    reason = "generator_exhausted"
    try:
        for knots in generate_knots(x, y, k=3, s=smoothing):
            kcount = len(knots) - 4
            if kcount > cap:
                reason = "control_point_cap"
                break
            if (np.any(np.diff(knots) < 0) or not np.all(knots[:4] == 0)
                    or not np.all(knots[-4:] == x[-1])):
                reason = "invalid_generated_knots"
                break
            # Ordered positive basis supports supply a Schoenberg-Whitney
            # matching, checking full column rank without a dense T-by-K SVD.
            basis = BSpline.design_matrix(x, knots, 3).tocsc()
            last_row = -1
            for column in range(kcount):
                start, stop = basis.indptr[column:column + 2]
                rows = basis.indices[start:stop]
                eligible = rows[(basis.data[start:stop] > 0) & (rows > last_row)]
                if not len(eligible):
                    raise ValueError("Generated knots violate independent positive support conditions")
                last_row = int(eligible[0])
            candidate = make_lsq_spline(x, y, knots, k=3, method="qr")
            candidate.extrapolate = False
            if not np.isfinite(candidate.c).all():
                reason = "nonfinite_coefficients"
                break
            spline = candidate
            error = float(np.abs(spline(x) - y).max())
            history.append(dict(control_points=kcount, max_abs_normalized_residual=error))
            if error <= tolerance:
                reason = "tolerance_met"
                break
    except (ValueError, np.linalg.LinAlgError) as exc:
        reason = f"solver_failure: {exc}"
    if spline is None:
        raise ValueError(f"No admissible initial fit: {reason}")
    residual = spline(x) - y
    return spline, dict(success=reason == "tolerance_met", reason=reason,
        max_abs_normalized_residual=float(np.abs(residual).max()),
        per_channel_max_abs_normalized_residual=np.abs(residual).max(axis=0).tolist(),
        per_channel_rmse_normalized=np.sqrt(np.mean(residual**2, axis=0)).tolist(),
        history=history, settings=dict(degree=3, tolerance=tolerance,
        max_control_points=cap, generator_s=smoothing, solver="make_lsq_spline/qr",
        insertion="scipy.generate_knots/FITPACK", scipy_version=scipy.__version__,
        endpoint_values_constrained=False))


def range_diagnostics(spline, actions, norm):
    """Exact per-channel cubic extrema, including between recorded samples."""
    from scipy.interpolate import PPoly
    extrema = []
    for j in range(actions.shape[1]):
        pp = PPoly.from_spline((spline.t, spline.c[:, j], spline.k))
        roots = pp.derivative().roots(extrapolate=False)
        roots = roots[np.isfinite(roots) & (roots >= 0) & (roots <= len(actions) - 1)]
        times = np.r_[0., len(actions) - 1., roots, np.unique(spline.t)]
        values = decode(spline(times), norm)[:, j]
        extrema.append([float(values.min()), float(values.max())])
    ranges = np.array(extrema)
    low, high = actions.min(axis=0), actions.max(axis=0)
    return dict(demonstrated_min=low.tolist(), demonstrated_max=high.tolist(),
                decoded_curve_min=ranges[:, 0].tolist(), decoded_curve_max=ranges[:, 1].tolist(),
                overshoot_below=np.maximum(low - ranges[:, 0], 0).tolist(),
                overshoot_above=np.maximum(ranges[:, 1] - high, 0).tolist())


def read_cache(path):
    with np.load(path, allow_pickle=False) as f:
        metadata = json.loads(str(f["metadata"]))
        actions = f["actions"].copy()
        spline = BSpline(f["knots"].copy(), f["control_points"].copy(), int(f["degree"]), extrapolate=False)
    return actions, spline, metadata


# Stage 2 contract. Scaling concerns knot times only; coefficient columns stay
# in the stage-1 normalized action coordinates, including continuous grippers.
LOCAL_SCHEMA = "motus-local-cubic-36-v1"


def validate_time_scaling(scaling):
    if scaling.get("scope") not in ("training-only", "diagnostic-only"):
        raise ValueError("Knot scaling must declare training-only or diagnostic-only scope")
    if scaling.get("units") != "source_action_step":
        raise ValueError("Knot scaling units must be source_action_step")
    if scaling["scope"] == "training-only":
        if ("identity" not in scaling or "normalization_identity" not in scaling
                or not scaling.get("training_sources")
                or scaling["identity"] != identity({k: v for k, v in scaling.items() if k != "identity"})):
            raise ValueError("Unverified training knot scaling identity/provenance")
    offset, scale = float(scaling["offset"]), float(scaling["scale"])
    if not np.isfinite([offset, scale]).all() or scale <= 0:
        raise ValueError("Invalid affine knot scaling")
    return offset, scale


def extract_local(spline, observation_time, scaling):
    """Slice a containing-span window; never refit, synthesize or shift a curve.

    36 knots and 32 effective coefficient rows, plus 4 structural rows. Missing
    tail knots/coefficients repeat their final values, explicitly marked. The
    decoder removes only surplus endpoint repetitions before evaluation.
    """
    offset, scale = validate_time_scaling(scaling)
    t, c, degree = spline.t, spline.c, spline.k
    obs = float(observation_time)
    if degree != 3 or c.ndim != 2 or c.shape[1] != 14 or len(c) != len(t) - 4:
        raise ValueError("Expected canonical unpadded cubic episode spline with 14 channels")
    if not np.isfinite(obs) or not t[3] <= obs <= t[-4]:
        raise ValueError("Observation outside episode spline domain")
    span = min(max(int(np.searchsorted(t, obs, side="right") - 1), 3), len(c) - 1)
    start = span - 3
    raw_knots = t[start:start + 36]
    n = len(raw_knots)
    active = n - 4
    if active < 4:
        raise ValueError("Insufficient local support")
    coefficients = c[start:start + 36]
    knots = np.pad(raw_knots, (0, 36 - n), mode="edge")
    cp = np.pad(coefficients, ((0, 36 - len(coefficients)), (0, 0)), mode="edge")
    parameters = np.column_stack(((knots - obs - offset) / scale, cp)).astype(np.float32)
    knot_sources = np.r_[np.arange(start, start + n), np.full(36 - n, -1)].astype(int)
    coefficient_sources = np.r_[np.arange(start, start + len(coefficients)), np.full(36 - len(coefficients), -1)].astype(int)
    domain = [float(raw_knots[3] - obs), float(raw_knots[-4] - obs)]
    return dict(schema=LOCAL_SCHEMA, degree=3, parameters=parameters, observation_time=obs,
                scaling=dict(scaling), span=span, global_start=start,
                real_knot_count=n, effective_control_points=active,
                knot_source_indices=knot_sources.tolist(), coefficient_source_indices=coefficient_sources.tolist(),
                structural_coefficient_rows=list(range(32, 36)),
                repeated_coefficient_rows=list(range(len(coefficients), 36)),
                valid_interval_relative=domain, future_interval_relative=[0., domain[1]],
                action_parameter_loss_mask="all rows supervised; padding is deterministic",
                knot_encoding="observation-relative absolute position; (time-offset)/scale")


def decode_local(parameters, timestamps, observation_time, scaling, normalization):
    """The shared deployment decoder. Timestamps are absolute source-row times.

    Reject inversions and unsupported coverage; never sort or extrapolate.
    Return applied structural trimming and raw continuous gripper range events.
    """
    offset, scale = validate_time_scaling(scaling)
    a = np.asarray(parameters, dtype=np.float64)
    if a.shape != (36, 15) or not np.isfinite(a).all():
        raise ValueError("Invalid packed parameters: expected finite (36,15)")
    with np.errstate(over="ignore", invalid="ignore"):
        u = a[:, 0] * scale + offset
    if not np.isfinite(u).all():
        raise ValueError("Nonfinite knots after inverse scaling")
    if np.any(np.diff(u) < 0):
        raise ValueError("Invalid knots: decreasing times; repair is disabled")
    n = len(u)
    # Explicit reference endpoint-padding rule: trim excess repeats at the right
    # only, along with the now-inactive coefficient rows. No arbitrary repair.
    while n > 8 and u[n - 5] == u[n - 1]:
        n -= 1
    u, cp = u[:n], a[:n - 4, 1:]
    _, counts = np.unique(u, return_counts=True)
    if np.any(counts > 4) or u[3] >= u[-4]:
        raise ValueError("Invalid knot multiplicity or degenerate decoding domain")
    if not np.isfinite(observation_time):
        raise ValueError("Nonfinite observation time")
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Controller timestamps must be finite ordered 1D values")
    relative = times - observation_time
    # Float32 packing changes knot coordinates slightly. Do not extrapolate:
    # permit only machine-roundoff boundary snaps and report every such snap.
    atol = 4 * np.finfo(np.float32).eps * max(1., np.max(np.abs(u)), scale)
    if relative.min() < u[3] - atol or relative.max() > u[-4] + atol:
        raise ValueError(f"Insufficient coverage: requested {relative[[0,-1]].tolist()}, domain {[u[3],u[-4]]}")
    evaluated = np.clip(relative, u[3], u[-4])
    values = BSpline(u, cp, 3, extrapolate=False)(evaluated)
    low, high = np.asarray(normalization["low"]), np.asarray(normalization["high"])
    if low.shape != (14,) or high.shape != (14,) or not np.isfinite([low, high]).all() or np.any(high < low):
        raise ValueError("Invalid action normalization bounds")
    commands = decode(values, normalization)
    if not np.isfinite(commands).all():
        raise ValueError("Nonfinite decoded commands")
    return commands, dict(valid_interval_relative=[float(u[3]), float(u[-4])],
        trimmed_tail_knots=36 - n, effective_control_points=n - 4,
        boundary_roundoff_snaps=int(np.count_nonzero(evaluated != relative)),
        gripper_out_of_range_count=int(np.count_nonzero((commands[:, [6, 13]] < 0) | (commands[:, [6, 13]] > 1))),
        gripper_policy="continuous unchanged; simulator retains its existing saturation",
        repaired_knots=False, extrapolated=False)


def fit_knot_time_scaling(caches):
    """Fit affine bounds from all integer observations in TRAINING caches only.

    All caches must share the recorded stage-1 normalization/split; membership
    and action hashes are checked. Call once with exactly the complete training
    cache set, then persist and share the returned object across splits.
    """
    identity_map = dict(scope="diagnostic-only", units="source_action_step", offset=0., scale=1.)
    bounds, seen, expected, norm_id = [], set(), None, None
    for path in caches:
        actions, spline, meta = read_cache(path)
        norm = meta["normalization"]
        if norm["scope"] != "training-only" or not meta["diagnostics"]["success"]:
            raise ValueError("Time statistics require successful fits and training-only normalization")
        # Paths distinguish genuinely identical trajectories in distinct episodes.
        current = {(s["path"], s["action_sha256"]) for s in norm["training_sources"]}
        if expected is None:
            expected, norm_id = current, norm["identity"]
        if current != expected or norm_id != norm["identity"]:
            raise ValueError("Inconsistent episode split/normalization")
        key = (meta["source"]["path"], meta["source"]["action_sha256"])
        if key not in expected or key in seen:
            raise ValueError("Validation, unknown or duplicate episode in time statistics")
        seen.add(key)
        for observation in range(len(actions)):
            sample = extract_local(spline, observation, identity_map)
            # Use original double precision knots, not packed float32 values.
            start = sample["global_start"]
            knots = spline.t[start:start + 36] - observation
            bounds.extend([float(knots.min()), float(knots.max())])
    if not seen or seen != expected:
        raise ValueError("Supply exactly all training episodes from the fixed split")
    low, high = min(bounds), max(bounds)
    result = dict(scope="training-only", units="source_action_step", offset=(low + high) / 2,
                  scale=max((high - low) / 2, 1.), training_bounds=[low, high],
                  normalization_identity=norm_id, training_sources=sorted(seen), schema=LOCAL_SCHEMA)
    result["identity"] = identity(result)
    return result


def select_future_frames(sample, action_times, camera_times, *, count=8, knot_offset=1, knot_stride=4):
    """Knot indices 3+offset+i*stride; nearest camera timestamp, earlier tie.

    Arrays must use the same explicitly declared clock (seconds on real data).
    Repeated tail/support requests are held at the last evaluable knot and
    masked. No inference from video FPS is made here.
    """
    action_times, camera_times = np.asarray(action_times, float), np.asarray(camera_times, float)
    if (action_times.ndim != 1 or camera_times.ndim != 1 or min(len(action_times), len(camera_times)) < 1
            or not np.isfinite(action_times).all() or not np.isfinite(camera_times).all()
            or np.any(np.diff(action_times) < 0) or np.any(np.diff(camera_times) < 0)):
        raise ValueError("Invalid action/camera timestamps")
    if count < 1 or count % 4 or knot_offset < 1 or knot_stride < 1:
        raise ValueError("Need positive knot offset/stride and future-frame count divisible by four")
    obs = sample["observation_time"]
    if not 0 <= obs <= len(action_times) - 1:
        raise ValueError("Observation outside action timeline")
    offset, scale = validate_time_scaling(sample["scaling"])
    knots = sample["parameters"][:, 0].astype(float) * scale + offset + obs
    # Effective right endpoint before float32 quantization is recorded by extractor.
    end = obs + sample["valid_interval_relative"][1]
    if end > len(action_times) - 1:
        raise ValueError("Local window exceeds the supplied action timeline")
    records, used = [], set()
    current_time = float(np.interp(obs, np.arange(len(action_times)), action_times))
    if not camera_times[0] <= current_time <= camera_times[-1]:
        raise ValueError("Current observation is outside camera timestamp coverage")
    def nearest(time):
        return int(np.argmin(np.abs(camera_times - time)))
    current_index = nearest(current_time)
    used.add(current_index)
    last_request = end
    for i in range(count):
        index = 3 + knot_offset + i * knot_stride
        supported = index <= 32 and index < sample["real_knot_count"] - 3
        request = float(knots[index]) if index < 36 else last_request
        # Roundoff at an episode endpoint is handled in the decoder similarly.
        request = min(max(request, obs), end)
        last_request = request
        timestamp = float(np.interp(request, np.arange(len(action_times)), action_times))
        frame = nearest(timestamp)
        outside_camera = timestamp < camera_times[0] or timestamp > camera_times[-1]
        repeat = frame in used
        valid = bool(supported and request > obs and not repeat and not outside_camera)
        records.append(dict(knot_row=index, global_knot_index=sample["global_start"] + index if supported else None,
            requested_action_time=request, target_timestamp=timestamp, camera_index=frame,
            camera_timestamp=float(camera_times[frame]), rounding_error=float(camera_times[frame] - timestamp),
            valid=valid, repeated_frame=repeat, support_or_tail_padding=not supported,
            outside_camera_clock=bool(outside_camera)))
        used.add(frame)
    raw_valid = [True] + [r["valid"] for r in records]
    # Causal caches let earlier invalid frames contaminate every later latent.
    # This conservative prefix rule avoids claiming exact receptive fields.
    latent_valid = [True] + [all(raw_valid[:1 + 4 * (i + 1)]) for i in range(count // 4)]
    return dict(current_camera_index=current_index, current_timestamp=current_time,
                current_camera_timestamp=float(camera_times[current_index]),
                current_rounding_error=float(camera_times[current_index] - current_time),
                frames=records, raw_frame_valid=raw_valid, conservative_latent_valid=latent_valid,
                proposed_latent_loss_mask=[False] + latent_valid[1:], proposed_latent_key_mask=latent_valid,
                unchanged_motus_training_compatible=all(raw_valid),
                mask_status="inspection-only; MOTUS currently consumes neither mask; padded samples blocked",
                settings=dict(count=count, knot_offset=knot_offset, knot_stride=knot_stride,
                              rounding="nearest timestamp, earlier index on ties", vae_future_group=4))


def decode_episode_grid(spline, times, scaling, normalization):
    """Decode a whole ordered timestamp grid using only the packed decoder."""
    times = np.asarray(times, float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Invalid episode decoding grid")
    outputs, reports, sample = [], [], None
    for time in times:
        if sample is None or time > sample["observation_time"] + sample["valid_interval_relative"][1]:
            sample = extract_local(spline, time, scaling)
        result, report = decode_local(sample["parameters"], [time], sample["observation_time"], scaling, normalization)
        outputs.append(result[0])
        reports.append(report)
    return np.asarray(outputs), reports
