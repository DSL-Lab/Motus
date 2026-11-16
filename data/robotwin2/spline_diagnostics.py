"""Score raw and safeguarded predictions through the deployment decoder."""
import numpy as np

from .adaptive_spline import decode_local, prepare_local


def score_spline_prediction(parameters, timestamps, observation_time, scaling, normalization, target):
    """Keep invalid command rows NaN; report executable prefix separately.

    Raw validity disables only running maximum. Both paths apply the same
    structural endpoint trimming and strict multiplicity/domain checks.
    """
    times = np.asarray(timestamps, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('Expected finite ordered execution timestamps')
    target = np.asarray(target, dtype=float)
    if target.shape != (len(times), 14) or not np.isfinite(target).all():
        raise ValueError('Expected finite target commands on the requested grid')
    commands = np.full(target.shape, np.nan)
    record = dict(raw_full_prefix_valid=False, repaired_spline_valid=False,
                  full_prefix_valid=False, covered_steps=0, supported_steps=0,
                  requested_steps=len(times), execution_prefix_coverage=0.,
                  decoded_rmse_per_channel=None, covered_prefix_rmse_per_channel=None,
                  supported_rmse_per_channel=None)
    try:
        decode_local(parameters, times, observation_time, scaling, normalization, repair_knots=False)
        record['raw_full_prefix_valid'] = True
    except ValueError as exc:
        record['raw_invalid_reason'] = str(exc)
    try:
        knots, _, safeguards = prepare_local(parameters, scaling)
        record['decoder_report'] = safeguards
        record.update({k: safeguards[k] for k in ('raw_knots_finite', 'raw_knots_monotone', 'raw_domain_relative')})
        record['raw_domain_valid'] = bool(safeguards['raw_domain_relative'][0] < safeguards['raw_domain_relative'][1] and
            safeguards['raw_domain_relative'][0] <= 0 <= safeguards['raw_domain_relative'][1])
        # Validate the whole repaired spline, independently of prefix coverage.
        decode_local(parameters, np.array([knots[3], knots[-4]]) + observation_time,
                     observation_time, scaling, normalization)
        record['repaired_spline_valid'] = True
        supported, errors = [], []
        for i, time in enumerate(times):
            try:
                value, _ = decode_local(parameters, [time], observation_time, scaling, normalization)
                supported.append(i)
                errors.append(value[0] - target[i])
                if i == record['covered_steps']:
                    commands[i] = value[0]
                    record['covered_steps'] += 1
            except ValueError:
                pass
        covered = record['covered_steps']
        record['supported_steps'] = len(supported)
        record['execution_prefix_coverage'] = covered / len(times)
        if errors:
            record['supported_rmse_per_channel'] = np.sqrt(np.mean(np.square(errors), axis=0)).tolist()
        if covered:
            record['covered_prefix_rmse_per_channel'] = np.sqrt(np.mean((commands[:covered] - target[:covered])**2, axis=0)).tolist()
        # A deployable chunk requires the complete requested execution prefix.
        full, report = decode_local(parameters, times, observation_time, scaling, normalization)
        commands = full
        record['decoder_report'] = report
        record['full_prefix_valid'] = True
        record['decoded_rmse_per_channel'] = np.sqrt(np.mean((commands - target)**2, axis=0)).tolist()
    except ValueError as exc:
        record['invalid_reason'] = str(exc)
    return commands, record
