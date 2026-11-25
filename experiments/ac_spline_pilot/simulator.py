"""Actual RoboTwin rollouts using its existing episode loop and A/C bundles."""
import argparse
from collections import Counter
import copy
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
from statistics import mean
import sys
import time
import traceback

from .rollout_policy import ROBOTWIN_ROOT, load_policy


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def robotwin_api():
    sys.path.insert(0, str(ROBOTWIN_ROOT / 'scripts'))
    return importlib.import_module('eval_policy_xpolicylab')


def evaluation_cases(api, args, path, *, episodes, seed, instruction_type, max_attempts):
    """Expert-filter once, then reuse the exact scenes AND instructions for A/C."""
    specification = dict(schema='motus-robotwin-cases-v1', task=args['task_name'],
        task_config=args['task_config'], episodes=episodes, seed=seed, instruction_type=instruction_type)
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text())
        if any(manifest.get(key) != value for key, value in specification.items()) or not manifest.get('complete'):
            raise ValueError('Existing simulator cases do not match the requested evaluation')
        if len(manifest['cases']) != episodes or len({c['seed'] for c in manifest['cases']}) != episodes:
            raise ValueError('Simulator cases must contain exactly the requested distinct seeds')
        return manifest
    started = time.perf_counter()
    cases, skipped = [], []
    task = api.class_decorator(args['task_name'])
    expert_args = dict(args, eval_mode=True, render_freq=0)
    expert_args.pop('eval_video_save_dir', None)
    for candidate in range(100000 * (1 + seed), 100000 * (1 + seed) + max_attempts):
        random.seed(candidate)
        try:
            task.setup_demo(now_ep_num=len(cases), seed=candidate, is_test=True, **expert_args)
            info = task.play_once()
            if not (task.plan_success and task.check_success()):
                skipped.append(dict(seed=candidate, reason='expert_failed'))
                continue
            instruction = api.build_instruction(args, info, instruction_type, episodes)
            cases.append(dict(seed=candidate, instruction=instruction))
        except Exception as exc:
            skipped.append(dict(seed=candidate, reason=f'{type(exc).__name__}: {exc}'))
        finally:
            task.close_env()
        if len(cases) == episodes:
            break
    manifest = dict(specification, complete=len(cases) == episodes, cases=cases, skipped=skipped,
                    selection_wall_seconds=time.perf_counter() - started, expert_check=True)
    write_json(path, manifest)
    if not manifest['complete']:
        raise RuntimeError(f'Found {len(cases)}/{episodes} expert-valid scenes in {max_attempts} attempts')
    return manifest


def evaluate_cases(api, policy, args, manifest, output):
    """Never substitute another seed when a policy/setup/decoder fails."""
    counter = mp.Value('i', 0)
    records = []
    videos = output / 'videos'
    video_size = None
    if args.get('eval_video_log'):
        videos.mkdir()
        args['eval_video_save_dir'] = str(videos)
        camera = api.get_camera_config(args['camera']['head_camera_type'])
        video_size = f"{camera['w']}x{camera['h']}"
    task = api.class_decorator(args['task_name'])
    with (output / 'episodes.jsonl').open('x') as stream:
        for index, case in enumerate(manifest['cases']):
            random.seed(case['seed'])
            policy.begin_episode(case['seed'], case['instruction'], episode_id=index)
            started = time.perf_counter()
            try:
                record = api.run_one_batch_episode(0, index, case['seed'], args['task_name'],
                    task, copy.deepcopy(args), policy, action_type='joint', frequency=30,
                    expert_check=False, video_size=video_size,
                    instruction_type=manifest['instruction_type'], test_num=len(manifest['cases']),
                    episode_counter=counter, instruction_override=case['instruction'])
                if record['type'] != 'episode_done':
                    raise RuntimeError(f'Unexpected result for fixed scene: {record}')
            except Exception as exc:
                traceback.print_exc()
                record = dict(type='episode_done', episode_id=index, seed=case['seed'],
                    success=False, official_success=None, failure_reason=f'{type(exc).__name__}: {exc}',
                    episode_wall_seconds=time.perf_counter() - started)
                task.close_env()
            record.setdefault('official_success', record['success'])
            record['instruction'] = case['instruction']
            record['calls'] = policy.calls
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            records.append(record)
            print(f"Simulator scene {index+1}/{len(manifest['cases'])}, seed={case['seed']}, success={record['success']}, reason={record.get('failure_reason')}", flush=True)
    return records


def summarize_episodes(records):
    def average(rows, field):
        values = [row[field] for row in rows if row.get(field) is not None]
        return mean(values) if values else None

    calls = [call for episode in records for call in episode['calls']]
    steps = [count for episode in records for count in episode.get('commands_per_call', [])]
    diagnostics = [call['spline_diagnostics'] for call in calls if 'spline_diagnostics' in call]
    knots = [d for d in diagnostics if d.get('raw_knots_finite')]
    failures = Counter(r['failure_reason'] for r in records if r.get('failure_reason'))
    categories = Counter(failure_category(r) for r in records if not r['success'])
    total = len(records)
    successes = sum(r['success'] for r in records)
    rate = successes / total
    z = 1.959963984540054
    center = (rate + z**2 / (2 * total)) / (1 + z**2 / total)
    width = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / (1 + z**2 / total)
    return dict(episodes=total, successes=successes, failures=total - successes, success_rate=rate,
        success_rate_wilson_95=[max(0., center - width), min(1., center + width)],
        failure_categories=dict(categories), failure_reasons=dict(failures),
        missing_official_result=sum(r['official_success'] is None for r in records),
        average_simulated_seconds=average(records, 'simulated_seconds'),
        simulated_duration_records=sum(r.get('simulated_seconds') is not None for r in records),
        average_episode_wall_seconds=average(records, 'episode_wall_seconds'),
        average_controller_wall_seconds=average(records, 'execution_wall_seconds'),
        average_policy_calls=sum(r.get('policy_calls', len(r['calls'])) for r in records) / total,
        average_commands_per_call=mean(steps) if steps else None,
        average_inference_wall_seconds=average(calls, 'inference_wall_seconds'),
        average_decode_wall_seconds=average(calls, 'decode_wall_seconds'),
        average_total_policy_wall_seconds=average(calls, 'total_wall_seconds'),
        timing_sample_counts={k: sum(k in call for call in calls) for k in
            ('inference_wall_seconds', 'decode_wall_seconds', 'total_wall_seconds')},
        spline_diagnostics=dict(calls=len(diagnostics), calls_with_finite_knots=len(knots),
            raw_knot_inversion_rate=mean(not d['raw_knots_monotone'] for d in knots) if knots else None,
            running_maximum_repair_frequency=average(knots, 'repaired_knots'),
            average_repair_displacement_source_rows=average(knots, 'repair_displacement'),
            invalid_multiplicity_count=sum('Invalid knot multiplicity' in d.get('exception', '') for d in diagnostics),
            temporal_coverage_failures=sum(d.get('coverage_valid') is False for d in diagnostics),
            decoder_failures=sum('decode_wall_seconds' in call and not call['decoded_valid'] for call in calls)
        ) if diagnostics else None)


def failure_category(record):
    if record['official_success'] is None:
        return 'episode_error_without_official_result'
    reason = record.get('failure_reason') or ''
    if reason == 'command_limit':
        return reason
    calls = record['calls']
    if calls and 'decode_wall_seconds' in calls[-1] and not calls[-1]['decoded_valid']:
        if 'Invalid knot multiplicity' in reason:
            return 'invalid_knot_multiplicity'
        if 'Insufficient coverage' in reason:
            return 'insufficient_spline_coverage'
        return 'spline_decode_error'
    return 'rollout_error'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', choices=['place_can_basket'], default='place_can_basket')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--case-prefix', action='store_true',
                        help='Use the first --episodes from an existing complete manifest (for smoke testing)')
    parser.add_argument('--variant', choices=['A', 'C'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-seed-attempts', type=int)
    parser.add_argument('--spline-diagnostics', action=argparse.BooleanOptionalAction, default=None,
                        help='Record per-call knot and decoder diagnostics in episodes.jsonl')
    parser.add_argument('--spline-execution-rows', type=int, choices=range(8, 37), metavar='8..36',
                        help='Override the spline rollout row budget for this evaluation')
    parser.add_argument('--check', action='store_true', help='Check the checkpoint contract without loading models or simulation')
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or (args.max_seed_attempts is not None and args.max_seed_attempts < args.episodes):
        parser.error('Need positive episodes, nonnegative seed and enough seed attempts')
    args.cases = args.cases.resolve()
    checkpoint = args.checkpoint.resolve(strict=True)
    for file in ('research_config.json', 'research_adapter.pt'):
        if not (checkpoint / file).is_file():
            raise FileNotFoundError(checkpoint / file)
    metadata = json.loads((checkpoint / 'research_config.json').read_text())
    contract = metadata['contract']
    from robotwin_spline import identity
    if contract['identity'] != identity({key: value for key, value in contract.items() if key != 'identity'}):
        raise ValueError('Checkpoint contract identity mismatch')
    if contract['dataset_split']['task'] != args.task or contract['state_dim'] != 14:
        raise ValueError('Checkpoint task/embodiment does not match this rollout')
    if args.variant and contract['representation'] != {'A': 'dense', 'C': 'spline'}[args.variant]:
        parser.error('Checkpoint representation does not match the selected variant')
    if args.spline_execution_rows is not None and contract['representation'] != 'spline':
        parser.error('--spline-execution-rows requires a spline checkpoint')
    if args.check:
        print(json.dumps(dict(checkpoint=str(checkpoint), task=args.task, representation=contract['representation'],
                             episodes=args.episodes, seed=args.seed, mode='simulator checkpoint check'), indent=2))
        return
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME') or not torch.cuda.is_available():
        raise RuntimeError('Simulator/model execution requires an allocated Slurm GPU; --check is CPU-only')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(evaluation='RoboTwin closed-loop simulator rollouts', task=args.task,
        checkpoint=str(checkpoint), contract_identity=contract['identity'], representation=contract['representation'],
        checkpoint_metadata_sha256=hashlib.sha256((checkpoint / 'research_config.json').read_bytes()).hexdigest(),
        requested_episodes=args.episodes, seed=args.seed, status='running', successes=None,
        dataset_split=contract['dataset_split'], executed_prefix=contract['executed_prefix'],
        cases=str(args.cases.resolve()), inference_steps=metadata['config']['model']['inference']['num_inference_timesteps'],
        gpu=torch.cuda.get_device_name(), timing_note='Timings include first-call overhead. Simulated time counts controller physics steps only.')
    started = time.perf_counter()
    write_json(output / 'results.json', report)
    original_cwd = Path.cwd()
    try:
        # RoboTwin resolves task assets and instruction templates from its root.
        os.chdir(ROBOTWIN_ROOT)
        api = robotwin_api()
        task_args, _ = api.load_task_args(dict(task_name=args.task, task_config='demo_clean',
            ckpt_setting=str(checkpoint), policy_name='Motus_AC'))
        task_args['eval_mode'] = True
        instruction_type = api.resolve_eval_instruction({}, task_args)
        if args.case_prefix:
            case_count = json.loads(args.cases.read_text())['episodes']
            if args.episodes > case_count:
                raise ValueError('Requested prefix exceeds the existing case manifest')
        else:
            case_count = args.episodes
        manifest = evaluation_cases(api, task_args, args.cases, episodes=case_count,
            seed=args.seed, instruction_type=instruction_type, max_attempts=args.max_seed_attempts or args.episodes * 50)
        manifest = dict(manifest, cases=manifest['cases'][:args.episodes])
        report['episode_seeds'] = [case['seed'] for case in manifest['cases']]
        report['cases_sha256'] = hashlib.sha256(args.cases.read_bytes()).hexdigest()
        settings = {k: v for k, v in task_args.items() if k != 'ckpt_setting'}
        report['simulator_settings_sha256'] = hashlib.sha256(
            json.dumps(dict(settings=settings, action_type='joint', frequency=30), sort_keys=True).encode()
        ).hexdigest()
        loading_started = time.perf_counter()
        policy = load_policy(checkpoint, [case['instruction'] for case in manifest['cases']])
        if args.spline_diagnostics is not None:
            policy.spline_diagnostics = args.spline_diagnostics
        if args.spline_execution_rows is not None:
            policy.execution_prefix = dict(policy.execution_prefix, row_budget=args.spline_execution_rows)
        if contract['representation'] == 'spline':
            report.update(executed_prefix=policy.execution_prefix, spline_diagnostics=policy.spline_diagnostics)
        report['policy_load_and_language_wall_seconds'] = time.perf_counter() - loading_started
        records = evaluate_cases(api, policy, task_args, manifest, output)
        report.update(summarize_episodes(records))
        report['status'] = 'incomplete' if report['missing_official_result'] else 'complete'
    except Exception as exc:
        report.update(status='error', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        write_json(output / 'results.json', report)
        os.chdir(original_cwd)
    if report['status'] == 'incomplete':
        raise SystemExit('One or more simulator episodes ended before an official result was available')


if __name__ == '__main__':
    main()
