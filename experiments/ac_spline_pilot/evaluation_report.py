"""Generate a Markdown report from matched offline and closed-loop A/C runs."""
import argparse
import json
from pathlib import Path
from statistics import mean


def read_pair(paths, match):
    a, c = [json.loads((path / 'results.json').read_text()) for path in paths]
    if [a['representation'], c['representation']] != ['dense', 'spline']:
        raise ValueError('Supply Variant A followed by Variant C')
    if any(r['status'] != 'complete' for r in (a, c)):
        raise ValueError('Both evaluation runs must be complete')
    for field in match:
        if a[field] != c[field]:
            raise ValueError(f'Runs differ in {field}')
    return a, c


def table(rows):
    return '\n'.join(['| Metric | Variant A | Variant C |', '|---|---:|---:|',
                      *[f'| {label} | {a} | {c} |' for label, a, c in rows]])


def value(number, scale=1, suffix=''):
    return 'Unavailable' if number is None else f'{number * scale:.4g}{suffix}'


def offline_table(results, group):
    def cell(result, field):
        stat = result['across_demonstrations'][group][field]
        if stat['mean'] is None:
            return 'Unavailable'
        pooled = value(result[group][field])
        return f"{pooled} ({value(stat['mean'])} ± {value(stat['std'])}, n={stat['demonstrations']})"

    rows = [(label, *(cell(r, field) for r in results)) for field, label in (
        ('ee_position_rmse_m', 'EE position RMSE (m)'),
        ('ee_orientation_mean_rad', 'Mean SO(3) orientation error (rad)'),
        ('joint_rmse_rad', 'Arm-joint RMSE (rad)'),
        ('gripper_rmse_fraction', 'Gripper RMSE (opening fraction)'),
    )]
    rows.append(('Full-window coverage', *(value(r['full_window_coverage'], 100, '%') for r in results)))
    rows.append(('Timestamp coverage', *(value(r['timestamp_coverage'], 100, '%') for r in results)))
    return table(rows)


def simulator_table(results):
    rows = [(label, *(value(r[field], scale, suffix) for r in results)) for label, field, scale, suffix in (
        ('Success rate', 'success_rate', 100, '%'),
        ('Successful episodes', 'successes', 1, ''),
        ('Failed episodes', 'failures', 1, ''),
        ('Episodes without official result', 'missing_official_result', 1, ''),
        ('Average simulated episode duration (s)', 'average_simulated_seconds', 1, ''),
        ('Average wall-clock episode duration (s)', 'average_episode_wall_seconds', 1, ''),
        ('Average controller wall time per episode (s)', 'average_controller_wall_seconds', 1, ''),
        ('Average policy calls per episode', 'average_policy_calls', 1, ''),
        ('Average commands executed per call', 'average_commands_per_call', 1, ''),
        ('Average inference latency (ms)', 'average_inference_wall_seconds', 1000, ''),
        ('Average total policy latency (ms)', 'average_total_policy_wall_seconds', 1000, ''),
        ('Wall-clock evaluation time (s)', 'wall_seconds', 1, ''),
    )]
    rows.insert(1, ('95% Wilson interval', *(
        ' to '.join(value(endpoint, 100, '%') for endpoint in r['success_rate_wilson_95']) for r in results)))
    rows.insert(-2, ('Average spline decoding latency (ms)', 'N/A',
                     value(results[1]['average_decode_wall_seconds'], 1000)))
    return table(rows)


def interpretation(offline, simulator, simulator_paths):
    a, c = simulator
    difference = 100 * (c['success_rate'] - a['success_rate'])
    lines = [
        f"The simulator recorded a C minus A success-rate difference of {difference:+.1f} percentage points "
        f"over {a['episodes']} shared cases. The Wilson intervals describe uncertainty for each rate.",
        'Offline errors measure imitation of demonstration targets. Task success comes from RoboTwin’s object-state evaluator.',
    ]
    if any(r['full_window_coverage'] < 1 for r in offline):
        lines.append('At least one policy has incomplete windows. The error tables use the coverage denominators shown and may reflect different subsets.')
    episode_maps = []
    for path in simulator_paths:
        records = [json.loads(line) for line in (path / 'episodes.jsonl').read_text().splitlines()]
        episode_maps.append({row['seed']: row for row in records if row['official_success'] is True})
    joined = episode_maps[0].keys() & episode_maps[1].keys()
    if joined:
        a_time, c_time = [mean(episodes[seed]['episode_wall_seconds'] for seed in joined)
                          for episodes in episode_maps]
        lines.append(f"Across {len(joined)} jointly successful cases, mean episode wall time was "
                     f"{a_time:.3g} s for A and {c_time:.3g} s for C.")
    else:
        lines.append('The runs have zero jointly successful cases, so episode wall time cannot establish a speed comparison for successful manipulation.')
    if a['gpu'] != c['gpu']:
        lines.append(f"GPU models differ ({a['gpu']} and {c['gpu']}), which confounds latency comparisons.")
    return '\n\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', nargs=2, type=Path, metavar=('A_DIR', 'C_DIR'), required=True)
    parser.add_argument('--simulator', nargs=2, type=Path, metavar=('A_DIR', 'C_DIR'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    offline = read_pair(args.offline, ('window_manifest_sha256', 'windows', 'demonstrations', 'requested_samples'))
    simulator = read_pair(args.simulator, ('cases_sha256', 'episode_seeds', 'simulator_settings_sha256',
                                            'dataset_split', 'inference_steps'))
    if any(o['checkpoint'] != s['checkpoint'] for o, s in zip(offline, simulator)):
        parser.error('Use the same checkpoint for offline scoring and simulator evaluation')
    notes = ['## Run details']
    for variant, result in zip(('A', 'C'), simulator):
        notes.extend([f"- Variant {variant} checkpoint `{result['checkpoint']}`",
                      f"- Variant {variant} recorded failure categories and counts "
                      f"`{json.dumps(result['failure_categories'], ensure_ascii=False)}`"])
    notes.append(f"- Variant C spline diagnostics `{json.dumps(simulator[1]['spline_diagnostics'])}`")
    sections = [
        '# MOTUS A/C evaluation on Place Can Basket',
        '## Offline results · complete windows',
        f"Across {offline[0]['demonstrations']} evaluated demonstrations. Each error cell shows "
        'the pooled metric followed by the mean ± population standard deviation of the per-demonstration metric.',
        offline_table(offline, 'full_window_errors'),
        '## Offline results · supported timestamps including partial windows',
        offline_table(offline, 'supported_errors'),
        '## Closed-loop RoboTwin results',
        simulator_table(simulator),
        '## Interpretation',
        interpretation(offline, simulator, args.simulator),
        notes[0] + '\n\n' + '\n'.join(notes[1:]),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        stream.write('\n\n'.join(sections) + '\n')


if __name__ == '__main__':
    main()
