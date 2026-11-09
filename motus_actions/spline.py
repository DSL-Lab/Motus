"""Timestamped, physical-unit B-splines. No policy, simulator, or logging imports.

Requires SciPy >= 1.15 (make_splprep supports >10 coordinated channels).
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_splprep


@dataclass(frozen=True)
class ActionSpline:
    degree: int
    knots: np.ndarray
    control_points: np.ndarray  # [number of control points, channels]
    source_domain: tuple[float, float]

    def __post_init__(self):
        k, t, c = self.degree, np.asarray(self.knots), np.asarray(self.control_points)
        lo, hi = self.source_domain
        if k != 3 or t.ndim != 1 or c.ndim != 2 or c.shape[1] == 0:
            raise ValueError("Expected a cubic spline with 2D control points")
        if len(t) != len(c) + k + 1 or len(c) < k + 1:
            raise ValueError("Inconsistent knot/control-point counts")
        if not all(np.isfinite(a).all() for a in (t, c, [lo, hi])):
            raise ValueError("Nonfinite spline parameters")
        if np.any(np.diff(t) < 0) or not lo < hi:
            raise ValueError("Invalid knots or source domain")
        if lo != t[k] or hi != t[-k - 1]:
            raise ValueError("Source domain must match the active knot domain")


class FitError(ValueError):
    def __init__(self, message, errors=None):
        super().__init__(message)
        self.errors = errors


def _samples(source_times, actions):
    t, y = np.asarray(source_times, dtype=float), np.asarray(actions, dtype=float)
    if t.ndim != 1 or y.ndim != 2 or len(t) != len(y) or len(t) < 2:
        raise ValueError("Expected timestamps [N] and actions [N, channels]")
    if not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Samples must be finite with strictly increasing timestamps")
    return t, y


def evaluate_spline(spline, source_times):
    """Decode parameters from any producer at source timestamps; never extrapolate."""
    t = np.asarray(source_times, dtype=float)
    lo, hi = spline.source_domain
    if not np.isfinite(t).all() or np.any(t < lo) or np.any(t > hi):
        raise ValueError("Evaluation timestamps outside the source domain")
    return BSpline(
        spline.knots, spline.control_points, spline.degree, extrapolate=False
    )(t)


def reconstruction_error(spline, source_times, actions):
    """Per-channel maximum absolute and RMS errors at the supplied timestamps."""
    t, y = _samples(source_times, actions)
    predicted = evaluate_spline(spline, t)
    if predicted.shape != y.shape:
        raise ValueError("Spline and reference channels differ")
    error = predicted - y
    return {
        "max_abs": np.max(np.abs(error), axis=0),
        "rms": np.sqrt(np.mean(error**2, axis=0)),
    }


def fit_spline(
    source_times, actions, tolerances, *, smoothing=1.0, max_control_points=None
):
    """Encode timestamped continuous actions, including demonstration actions.

    Divide channels by explicit physical-unit tolerances only during fitting;
    returned coefficients stay in physical units on the ORIGINAL source timeline.
    `smoothing` bounds normalized summed squared residuals, NOT maximum error.
    `max_control_points` is a hard output budget (SciPy's nest is only a target).
    """
    t, y = _samples(source_times, actions)
    tol = np.asarray(tolerances, dtype=float)
    if (
        len(t) < 4
        or tol.shape != (y.shape[1],)
        or not np.isfinite(tol).all()
        or np.any(tol <= 0)
    ):
        raise ValueError(
            "Cubic fitting needs >=4 samples and one positive tolerance per channel"
        )
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing must be finite and nonnegative")
    if max_control_points is not None and (
        not isinstance(max_control_points, int) or max_control_points < 4
    ):
        raise ValueError("max_control_points must be an integer >=4")
    # Do not use chord-length parameterization: actions are explicitly timestamped.
    nest = (
        None if max_control_points is None else min(max_control_points + 4, len(t) + 4)
    )
    # SciPy disallows nest for the s=0 interpolation special case. Check budget below.
    fitted, _ = make_splprep(
        (y / tol).T, u=t, k=3, s=smoothing, nest=nest if smoothing else None
    )
    result = ActionSpline(
        3, fitted.t.copy(), np.asarray(fitted.c) * tol, (float(t[0]), float(t[-1]))
    )
    errors = reconstruction_error(result, t, y)
    if (
        max_control_points is not None
        and len(result.control_points) > max_control_points
    ):
        raise FitError("Control-point budget exceeded", errors)
    if not np.isfinite(errors["max_abs"]).all() or np.any(errors["max_abs"] > tol):
        raise FitError("Maximum reconstruction tolerance exceeded", errors)
    return result


def save_spline(path, spline, source_times):
    """Save portable numeric parameters and source timestamps (no Python pickle)."""
    times = np.asarray(source_times, dtype=float)
    evaluate_spline(spline, times)  # validate the saved timestamps
    np.savez(
        path,
        degree=spline.degree,
        knots=spline.knots,
        control_points=spline.control_points,
        source_domain=spline.source_domain,
        source_times=times,
    )


def load_spline(path):
    with np.load(path, allow_pickle=False) as data:
        spline = ActionSpline(
            int(data["degree"]),
            data["knots"],
            data["control_points"],
            tuple(data["source_domain"]),
        )
        times = data["source_times"]
    evaluate_spline(spline, times)
    return spline, times
