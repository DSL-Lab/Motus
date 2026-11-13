"""Cubic episode fits and fixed local targets; see docs/spline_method_spec.md.

Times are source sample indices. Control points are never controller commands.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline

DEGREE = 3
ROWS = 36
VIDEO_ROWS = np.arange(4, 33, 4)
VERSION = "cubic-absolute-containing-v1"


def _finite(value, name):
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"nonfinite {name}")
    return value


@dataclass
class AffineScale:
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self):
        self.low = _finite(self.low, "normalization low")
        self.high = _finite(self.high, "normalization high")
        if self.low.shape != self.high.shape or np.any(self.high < self.low):
            raise ValueError("invalid normalization bounds")

    @property
    def width(self):
        # Constant channels map to zero, with invertible unit scale off-data.
        return np.where(self.high > self.low, self.high - self.low, 2.0)

    def encode(self, value):
        return 2 * (_finite(value, "values") - (self.low + self.high) / 2) / self.width

    def decode(self, value):
        return _finite(value, "parameters") * self.width / 2 + (self.low + self.high) / 2


def fit_episode(actions, tolerance=0.01):
    """Adaptive least squares on normalized actions; fail if bound is unmet.

    Knot insertion chooses the largest residual among unoccupied interior samples.
    No changes to raw actions and no error-tolerance fallback.
    """
    actions = _finite(actions, "actions")
    if actions.ndim != 2 or len(actions) < 4 or not 0 < tolerance < 1:
        raise ValueError("need >=4 action samples and tolerance in (0,1)")
    times = np.arange(len(actions), dtype=np.float64)
    interior = []
    while True:
        knots = np.r_[np.zeros(4), sorted(interior), np.full(4, times[-1])]
        curve = make_lsq_spline(times, actions, knots, k=DEGREE)
        residual = np.abs(curve(times) - actions).max(axis=1)
        if residual.max() <= tolerance:
            return curve, float(residual.max())
        if len(curve.c) >= len(times):
            raise ValueError(f"fit tolerance unmet: {residual.max()} > {tolerance}")
        candidates = np.arange(2, len(times) - 2)
        candidates = candidates[~np.isin(candidates, interior)]
        if not len(candidates):
            raise ValueError(f"no admissible knot insertion; error={residual.max()}")
        interior.append(float(candidates[np.argmax(residual[candidates])]))


def extract_local(curve, observation):
    """Return [36,1+D] absolute-time rows relative to current observation."""
    if curve.k != DEGREE or not curve.t[DEGREE] <= observation < curve.t[-4]:
        raise ValueError("observation outside global evaluable spans")
    span = int(np.searchsorted(curve.t, observation, side="right") - 1)
    start = min(max(span - DEGREE, 0), len(curve.t) - 8)
    knots = curve.t[start:start + ROWS]
    controls = curve.c[start:start + ROWS]
    knots = np.pad(knots, (0, ROWS - len(knots)), mode="edge")
    controls = np.pad(controls, ((0, ROWS - len(controls)), (0, 0)), mode="edge")
    return np.column_stack((knots - observation, controls))


def video_indices(local, observation, episode_length):
    """Eight offline knot targets. Duplicate/padded targets are ineligible."""
    local = _finite(local, "local spline")
    if local.shape[0] != ROWS:
        raise ValueError("expected 36 spline rows")
    indices = np.floor(observation + local[VIDEO_ROWS, 0] + 0.5).astype(np.int64)
    if (indices[0] <= observation or indices[-1] >= episode_length
            or np.any(np.diff(indices) <= 0)):
        raise ValueError("padded, duplicate or nonfuture video target")
    return indices


def local_curve(parameters):
    """Validate and construct supported curve, trimming structural tail padding."""
    parameters = _finite(parameters, "spline prediction")
    if parameters.ndim != 2 or parameters.shape[0] != ROWS or parameters.shape[1] < 2:
        raise ValueError("expected [36,1+D] spline parameters")
    knots = parameters[:, 0]
    if np.any(np.diff(knots) < 0):
        raise ValueError("invalid knot ordering")
    low, high = knots[3], knots[-4]
    if high <= low:
        raise ValueError("degenerate spline domain")
    # Offline windows repeat the final knot to fill rows. Remove surplus tail
    # repetitions together with the corresponding ineffective controls.
    end = len(knots)
    while end > 8 and knots[end - 5] == knots[-1]:
        end -= 1
    knots = knots[:end]
    values, counts = np.unique(knots, return_counts=True)
    # A cubic permits degree+1 multiplicity (a discontinuity). The chosen
    # continuous-window convention admits at most degree in the interior;
    # degree+1 is reserved for clamped boundaries. Running-max plateaus do not
    # become new boundaries, and are never trimmed out of the interior.
    interior = (values > knots[3]) & (values < knots[-4])
    if np.any(counts > 4) or np.any(counts[interior] > DEGREE):
        raise ValueError("invalid boundary/interior knot multiplicity")
    curve = BSpline(knots, parameters[:end - 4, 1:], DEGREE, extrapolate=False)
    return curve


def decode(parameters, controller_offsets):
    """Evaluate at matched controller offsets; never extrapolate or repair."""
    curve = local_curve(parameters)
    times = _finite(controller_offsets, "controller times")
    if times.ndim != 1 or not len(times) or np.any(np.diff(times) <= 0):
        raise ValueError("controller times must be increasing")
    if times[0] < curve.t[3] or times[-1] > curve.t[-4]:
        raise ValueError("insufficient execution coverage")
    return _finite(curve(times), "decoded controller commands").astype(np.float32)


def controller_commands(actions, arm_limits):
    """Validate native 14-D qpos commands and count native gripper saturation.

    RoboTwin robot.py:set_gripper clips opening fractions to [0,1]. Apply that
    existing controller behavior explicitly for both A/C, without arm clipping.
    """
    actions = _finite(actions, 'controller commands')
    limits = np.asarray(arm_limits, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or limits.shape != (12, 2):
        raise ValueError('expected 14-D commands and 12 native arm limits')
    if np.isnan(limits).any() or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError('invalid native joint limits')
    arm = actions[:, np.r_[0:6, 7:13]]
    if np.any(arm < limits[:, 0] - 1e-6) or np.any(arm > limits[:, 1] + 1e-6):
        raise ValueError('controller command outside native arm limits')
    grippers = actions[:, [6, 13]]
    clipped = np.clip(grippers, 0, 1)
    repairs = int(np.count_nonzero(clipped != grippers))
    result = actions.copy()
    result[:, [6, 13]] = clipped
    return result, repairs


def repair_prediction_knots(parameters):
    """SplineWAM Appendix D running maximum, with a scalar repair count.

    Preserve row/control-point order. Nonfinite inputs still reject; subsequent
    strict decoding rejects degenerate support or excessive interior multiplicity.
    Never apply this to offline fitted targets or use it to hide extraction errors.
    """
    parameters = _finite(parameters, 'spline prediction')
    if parameters.ndim != 2 or parameters.shape[0] != ROWS or parameters.shape[1] < 2:
        raise ValueError('expected [36,1+D] spline parameters')
    repaired = parameters.copy()
    repaired[:, 0] = np.maximum.accumulate(parameters[:, 0])
    count = int(np.count_nonzero(repaired[:, 0] != parameters[:, 0]))
    return repaired, count
