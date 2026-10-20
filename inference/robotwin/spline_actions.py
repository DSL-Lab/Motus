"""Frozen-policy postprocessing for RoboTwin's 14-D absolute joint targets.

Times are explicit source timestamps, not video-frame or latent indices. This
module does not change a simulator clock or provide a low-level controller.
"""

from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

JOINTS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPERS = (6, 13)


@dataclass
class Fit:
    spline: BSpline | None
    max_error: np.ndarray
    reason: str | None


def fit_cubic(times, values, tolerances, max_controls=None):
    """Adapt a shared clamped knot vector using the worst scaled residual.

    Least squares uses one design matrix for all coordinated channels. At the
    full sample budget SciPy's not-a-knot interpolant supplies a final candidate.
    Every candidate is checked at *all* original timestamps. A failed fit is
    never returned as an acceptable reconstruction.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    tolerances = np.broadcast_to(tolerances, (values.shape[1],)).astype(float)
    if not np.all(np.isfinite(tolerances)) or np.any(tolerances <= 0):
        raise ValueError("Per-channel tolerances must be finite and positive")
    if times.shape != (len(values),) or not np.all(np.diff(times) > 0):
        raise ValueError("Source timestamps must be strictly increasing")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(times)):
        raise ValueError("Nonfinite source trajectory")
    error = np.full(values.shape[1], np.inf)
    budget = min(len(values), len(values) if max_controls is None else max_controls)
    if budget < 4:
        return Fit(None, error, "fewer than four samples/control points")
    interior = []
    reason = "control-point budget exhausted"
    while len(interior) + 4 <= budget:
        try:
            if len(interior) + 4 == len(values):
                spline = make_interp_spline(times, values, k=3)
            else:
                knots = np.r_[[times[0]] * 4, interior, [times[-1]] * 4]
                basis = BSpline.design_matrix(times, knots, 3).toarray()
                coefficients, _, rank, _ = np.linalg.lstsq(basis, values, rcond=None)
                if rank != basis.shape[1]:
                    return Fit(None, error, "rank-deficient spline basis")
                spline = BSpline(knots, coefficients, 3, extrapolate=False)
            residual = np.abs(spline(times) - values)
            error = residual.max(axis=0)
            if np.all(np.isfinite(error)) and np.all(error <= tolerances):
                return Fit(spline, error, None)
            # Insert a common temporal knot, prioritizing the worst channel.
            scores = (residual / tolerances).max(axis=1)
            available = [
                i for i in range(1, len(times) - 1) if times[i] not in interior
            ]
            if not available:
                break
            index = max(available, key=lambda i: scores[i])
            interior = sorted([*interior, times[index]])
        except (ValueError, np.linalg.LinAlgError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            break
    return Fit(None, error, reason)


def source_schedule(times, speed_factor):
    """Map fixed-rate execution slots to source time, retaining both endpoints.

    The first predicted target is submitted in slot zero, as in dense Motus.
    The final slot clamps to the original prefix endpoint; it never extrapolates
    or extends the source prefix to fill an accelerated chunk. Endpoint rounding
    can add less than one execution period. Physical validity requires an
    executor with this fixed target period (RoboTwin TOPP does not have one).
    """
    times = np.asarray(times, dtype=np.float64)
    if speed_factor not in (1.0, 1.5, 2.0):
        raise ValueError("speed_factor must be 1.0, 1.5, or 2.0")
    if len(times) < 2 or not np.all(np.isfinite(times)):
        raise ValueError("Need at least two finite timestamps")
    periods = np.diff(times)
    if np.any(periods <= 0) or not np.allclose(
        periods, periods[0], rtol=1e-7, atol=1e-12
    ):
        raise ValueError("Fixed-rate source timestamps required")
    if speed_factor == 1.0:
        return times.copy()
    intervals = int(np.ceil((len(times) - 1) / speed_factor))
    source_indices = np.minimum(np.arange(intervals + 1) * speed_factor, len(times) - 1)
    # Snap integer source positions to their exact supplied timestamp. Repeated
    # floating point period arithmetic could otherwise delay a gripper change.
    return np.interp(source_indices, np.arange(len(times)), times)


class ActionAdapter:
    """Cubic reconstruction with analog gripper targets held separately.

    Only absolute bounded joint coordinates are supported. There is deliberately
    no generic scalar interpolation of rotations, deltas, or velocity commands.
    Grippers retain their original open-fraction values, without thresholds or
    clipping. Zero-order hold moves each change to the first mapped source time
    at/after its timestamp; multiple changes inside a slot are logged as skipped.
    """

    def __init__(
        self,
        mode="dense",
        speed_factor=1.0,
        joint_tolerances=0.01,
        max_controls=16,
        action_semantics="absolute_joint_position",
    ):
        if mode not in ("dense", "spline"):
            raise ValueError("mode must be dense or spline")
        if speed_factor not in (1.0, 1.5, 2.0):
            raise ValueError("Unsupported speed factor")
        if action_semantics != "absolute_joint_position":
            raise ValueError("Only RoboTwin absolute joint-position actions supported")
        self.mode = mode
        self.speed_factor = speed_factor
        self.tolerances = np.broadcast_to(joint_tolerances, (12,)).astype(float)
        if not np.all(np.isfinite(self.tolerances)) or np.any(self.tolerances <= 0):
            raise ValueError("Need twelve positive finite joint tolerances (radians)")
        if not isinstance(max_controls, int) or max_controls < 4:
            raise ValueError("max_controls must be an integer >= 4")
        self.max_controls = max_controls

    def adapt(self, actions, timestamps, executed_prefix=None):
        started = perf_counter()
        actions = np.asarray(actions)
        times = np.asarray(timestamps, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) != len(times):
            raise ValueError("Expected [N,14] actions and N source timestamps")
        if not np.issubdtype(actions.dtype, np.floating) or not np.all(
            np.isfinite(actions)
        ):
            raise ValueError(
                "Need finite floating point actions; nonfinite dense actions are unsafe"
            )
        prefix = len(actions) if executed_prefix is None else executed_prefix
        if not isinstance(prefix, int) or not 2 <= prefix <= len(actions):
            raise ValueError("Executed prefix must contain 2..N samples")
        # Fit the full model prediction, but execute only the original prefix.
        source_schedule(times, 1.0)
        query = source_schedule(times[:prefix], self.speed_factor)
        fit = None
        if self.mode == "spline":
            fit = fit_cubic(
                times, actions[:, JOINTS], self.tolerances, self.max_controls
            )
            if fit.spline is not None:
                reconstructed = fit.spline(times).astype(actions.dtype)
                fit.max_error = np.abs(
                    reconstructed.astype(float) - actions[:, JOINTS]
                ).max(axis=0)
                if not np.all(np.isfinite(fit.max_error)) or np.any(
                    fit.max_error > self.tolerances
                ):
                    fit.spline = None
                    fit.reason = "reconstruction exceeds tolerance after controller dtype conversion"
        if self.mode == "dense" and self.speed_factor == 1.0:
            output = actions[
                :prefix
            ].copy()  # bit-for-bit dense baseline, no interpolation
        else:
            output = np.empty((len(query), 14), dtype=actions.dtype)
            for channel in JOINTS:
                output[:, channel] = np.interp(query, times, actions[:, channel])
            # Exact original analog values; both dense and spline use this map.
            indices = np.searchsorted(times, query, side="right") - 1
            output[:, GRIPPERS] = actions[indices][:, GRIPPERS]
            if fit is not None and fit.spline is not None:
                candidate = fit.spline(query).astype(actions.dtype)
                # A finite source reconstruction can still overflow between samples.
                if np.all(np.isfinite(candidate)):
                    output[:, JOINTS] = candidate
                else:
                    fit.reason = "nonfinite decoded samples"
                    fit.spline = None
        if fit is not None and fit.spline is None and self.speed_factor == 1.0:
            output = actions[:prefix].copy()
        spline = fit.spline if fit is not None else None
        # Representation payload uses float64 knots/coefficients and original
        # gripper dtype. Source timestamps/layout are shared metadata. Expanded
        # controller actions and model activations are NOT compressed storage.
        dense_bytes = actions.nbytes
        payload_bytes = (
            spline.t.nbytes + spline.c.nbytes + actions[:, GRIPPERS].nbytes
            if spline is not None
            else dense_bytes
        )
        selected = np.searchsorted(times, query, side="right") - 1
        changes = (
            np.flatnonzero(
                np.any(np.diff(actions[:prefix, GRIPPERS], axis=0) != 0, axis=1)
            )
            + 1
        )
        collapsed = sum(
            max(0, int(np.count_nonzero((changes > a) & (changes <= b))) - 1)
            for a, b in pairwise(selected)
        )
        info = {
            "mode": self.mode,
            "speed_factor": self.speed_factor,
            "predicted_samples": len(actions),
            "prefix_samples": prefix,
            "output_samples": len(output),
            "source_span": [float(times[0]), float(times[prefix - 1])],
            "source_query": query.tolist(),
            "fallback": bool(fit is not None and spline is None),
            "fit_failure": fit.reason if fit is not None else None,
            "max_error_per_joint": (
                [float(e) if np.isfinite(e) else None for e in fit.max_error]
                if fit is not None
                else [0.0] * 12
            ),
            "control_points": len(spline.c) if spline is not None else None,
            "dense_bytes": dense_bytes,
            "representation_bytes": payload_bytes,
            "storage_ratio_dense_over_representation": dense_bytes / payload_bytes,
            "collapsed_gripper_changes": collapsed,
            "processing_s": perf_counter() - started,
        }
        return output, info


def boundary_jumps(previous, current, period_s=None):
    """Command jumps only. Velocities require a real, uniform target period."""
    if previous is None or len(previous) == 0 or len(current) == 0:
        return {}
    result = {
        "position_jump_per_joint": (current[0, JOINTS] - previous[-1, JOINTS]).tolist()
    }
    if period_s is not None and len(previous) >= 2 and len(current) >= 2:
        incoming = (current[1, JOINTS] - current[0, JOINTS]) / period_s
        outgoing = (previous[-1, JOINTS] - previous[-2, JOINTS]) / period_s
        result["velocity_jump_per_joint"] = (incoming - outgoing).tolist()
    return result
