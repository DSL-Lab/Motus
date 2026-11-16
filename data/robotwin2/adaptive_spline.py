"""MOTUS 36-row targets, time scaling, video selection and deployment adapter.

Shared episode math and sidecars live in the installed robotwin_spline package.
Import generic utilities from robotwin_spline directly.
See RESEARCH_TRAINING.md for setup.
"""
import numpy as np
from robotwin_spline import identity, read_cache
from robotwin_spline.local import extract_segment, evaluate_local

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
    segment = extract_segment(spline, obs, knot_count=36)
    span, start = segment["span"], segment["global_start"]
    raw_knots = segment["knots"]
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


def prepare_local(parameters, scaling, *, repair_knots=True):
    """Invert absolute knot scaling and apply Appendix D decoding safeguards.

    Running maximum preserves row order. Removing a leading surplus endpoint
    knot removes the leading inactive coefficient; trailing trimming removes
    trailing inactive coefficients. Interior multiplicities remain unchanged
    and are checked by the shared strict evaluator.
    """
    offset, scale = validate_time_scaling(scaling)
    a = np.asarray(parameters, dtype=np.float64)
    if a.shape != (36, 15) or not np.isfinite(a).all():
        raise ValueError("Invalid packed parameters: expected finite (36,15)")
    with np.errstate(over="ignore", invalid="ignore"):
        raw = a[:, 0] * scale + offset
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite knots after inverse scaling")
    if not repair_knots and np.any(np.diff(raw) < 0):
        raise ValueError("Invalid knots: decreasing times; repair is disabled")
    u = np.maximum.accumulate(raw) if repair_knots else raw.copy()
    delta = u - raw
    start, end = 0, len(u)
    # Cubic endpoints admit four repetitions. Keep len(u) == len(cp) + 4
    # and the original coefficient association after either boundary trim.
    while end - start > 8 and u[start] == u[start + 4]:
        start += 1
    while end - start > 8 and u[end - 5] == u[end - 1]:
        end -= 1
    knots, cp = u[start:end], a[start:end - 4, 1:]
    report = dict(repaired_knots=bool(np.any(delta)),
        repaired_knot_count=int(np.count_nonzero(delta)),
        knot_repair_max_source_steps=float(delta.max()),
        knot_repair_mean_source_steps=float(delta.mean()),
        knot_repair_max_normalized=float(delta.max() / scale),
        raw_knots_finite=True, raw_knots_monotone=bool(np.all(np.diff(raw) >= 0)),
        raw_domain_relative=[float(raw[3]), float(raw[-4])],
        valid_interval_relative=[float(knots[3]), float(knots[-4])],
        trimmed_head_knots=start, trimmed_tail_knots=36 - end,
        decoding_policy="absolute knots; running maximum; trim surplus endpoints; no extrapolation")
    return knots, cp, report


def decode_local(parameters, timestamps, observation_time, scaling, normalization, *, repair_knots=True):
    """Shared offline/deployment decoder at absolute source-row timestamps.

    Repair inversions with running maximum (Appendix D). Reject nonfinite
    values, invalid interior multiplicities and unsupported execution coverage.
    ``repair_knots=False`` measures raw prediction validity only.
    """
    u, cp, safeguards = prepare_local(parameters, scaling, repair_knots=repair_knots)
    commands, report = evaluate_local(u, cp, timestamps, observation_time,
                                      normalization, boundary_scale=float(scaling["scale"]))
    return commands, dict(report, **safeguards,
        gripper_out_of_range_count=int(np.count_nonzero((commands[:, [6, 13]] < 0) | (commands[:, [6, 13]] > 1))),
        gripper_policy="continuous unchanged; simulator retains its existing saturation")


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
