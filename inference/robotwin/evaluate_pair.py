"""RoboTwin task success for original MOTUS and matched A/C checkpoints.

Uses MotusPolicy/eval and RoboTwin's existing reset/fixed-rate controller. GPU and
simulator imports are lazy so plan and result checks can run on a login node.
Original runs independently; A/C use a shared fixed-rate execution contract.
"""
import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / 'WAM-RoboTwin'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_plan(plan, manifest, demonstration_seeds):
    """Never obtain test seeds from training/validation observations or outcomes."""
    if len(demonstration_seeds) != len(manifest['episodes']) or len(set(demonstration_seeds)) != len(demonstration_seeds):
        raise ValueError('authoritative demonstration seeds must match the episode count and be distinct')
    if plan.get('dataset_identity') != manifest['identity']:
        raise ValueError('test plan is not bound to this dataset manifest')
    if plan.get('task_name') not in {e['task_name'] for e in manifest['episodes']}:
        raise ValueError('test task differs from the demonstration task')
    if {e['task_name'] for e in manifest['episodes']} != {plan['task_name']}:
        raise ValueError('expected a single-task dataset')
    if plan.get('task_config') not in ('demo_clean', 'demo_randomized'):
        raise ValueError('unsupported task configuration')
    period = plan.get('control_dt_s')
    if isinstance(period, bool) or not isinstance(period, (int, float)) or not math.isfinite(period) or period <= 0:
        raise ValueError('test plan requires a verified positive controller period in seconds')
    if not isinstance(plan.get('clock_provenance'), str) or not plan['clock_provenance'].strip():
        raise ValueError('test plan requires controller-clock provenance; do not guess Hz')
    cases = plan.get('cases', [])
    if not cases:
        raise ValueError('declare at least one scene before evaluation')
    seeds = []
    for case in cases:
        if isinstance(case.get('seed'), bool) or not isinstance(case.get('seed'), int) or case['seed'] < 0:
            raise ValueError('each test scene needs a nonnegative integer seed')
        if not isinstance(case.get('instruction'), str) or not case['instruction'].strip():
            raise ValueError('each test scene needs a fixed, nonempty instruction')
        policy_seed = case.get('policy_seed', case['seed'])
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int) or policy_seed < 0:
            raise ValueError('policy seeds must be nonnegative integers')
        seeds.append(case['seed'])
    if len(set(seeds)) != len(seeds) or set(seeds) & set(demonstration_seeds):
        raise ValueError('test scenes must be distinct and disjoint from all demonstration scenes')
    return cases


def summarize(records, expected):
    """All fixed scenes remain in the denominator, including errors/timeouts."""
    if len({r['seed'] for r in records}) != len(records):
        raise ValueError('duplicate scene results')
    if len(records) > expected:
        raise ValueError('more records than planned scenes')
    if expected <= 0:
        raise ValueError('expected scene count must be positive')
    successes = sum(r['status'] == 'success' for r in records)
    counts = {}
    for row in records:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    complete = len(records) == expected
    interval = None
    if complete:
        p = successes / expected
        z = 1.959963984540054
        denominator = 1 + z*z/expected
        center = (p + z*z/(2*expected))/denominator
        half = z*math.sqrt(p*(1-p)/expected + z*z/(4*expected*expected))/denominator
        interval = [max(0., center-half), min(1., center+half)]
    timings = [t for row in records for t in row.get('policy_timings', []) if not t['warmup']]
    def quantile(values, fraction):
        if not values:
            return None
        values = sorted(values)
        index = (len(values)-1)*fraction
        low = int(index)
        return values[low] + (values[min(low+1,len(values)-1)]-values[low])*(index-low)
    latency = [t['observation_to_executable_action_s'] for t in timings]
    decoding = [t['spline_decode_s'] for t in timings]
    return dict(policy_timing_samples=len(timings), warmup_rule='exclude first policy call of each scene',
                policy_latency_median_s=quantile(latency,.5), policy_latency_p95_s=quantile(latency,.95),
                spline_decode_median_s=quantile(decoding,.5),
                status='completed' if complete else 'incomplete', planned_scenes=expected,
                recorded_scenes=len(records), successes=successes, outcomes=counts,
                success_rate=successes/expected if complete else None,
                wilson_95=interval, uncertainty='conditional on this checkpoint and fixed scene suite',
                repairs=sum(r.get('counts', {}).get('repairs', 0) for r in records),
                fallbacks=sum(r.get('counts', {}).get('fallbacks', 0) for r in records))


def measured_state(robot):
    """Read physical joints and invert native finger-qpos opening calibration.

    Adapted from tests/allocated_replay.py:measured_state; native 'real' getters
    still return commanded gripper openings, so those columns need replacement.
    """
    values = list(robot.get_left_arm_real_jointState() + robot.get_right_arm_real_jointState())
    for side, column in [('left',6), ('right',13)]:
        entity = getattr(robot, f'{side}_entity')
        joint, multiplier, bias = getattr(robot, f'{side}_gripper')[0]
        qpos = entity.get_qpos()[entity.get_active_joints().index(joint)]
        low, high = getattr(robot, f'{side}_gripper_scale')
        values[column] = float(((qpos-bias)/multiplier-low)/(high-low))
    return values


class SceneTrace:
    """Delegate scene operations and record each actual physics step after reset.

    Same scene-step interception as allocated_replay.MeasuredScene, with streamed
    telemetry to avoid retaining a full episode in memory. Robot/physics unchanged.
    """
    def __init__(self, scene, robot, stream):
        self.scene, self.robot, self.stream = scene, robot, stream
        self.steps, self.elapsed = 0, 0.
        self.started = time.perf_counter()

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def step(self):
        dt = float(self.scene.get_timestep())
        self.scene.step()
        self.steps += 1
        self.elapsed += dt
        robot = self.robot
        row = dict(physics_step=self.steps, simulated_time_s=self.elapsed,
                   wall_time_s=time.perf_counter()-self.started,
                   frame='simulator world', position_unit='metres', orientation='wxyz quaternion',
                   left_ee=robot.get_left_ee_pose(), right_ee=robot.get_right_ee_pose(),
                   measured_joints_and_grippers=measured_state(robot),
                   drive_targets=robot.get_left_arm_jointState()+robot.get_right_arm_jointState())
        self.stream.write(json.dumps(row, default=lambda x:x.tolist())+'\n')


def validate_matched_configs(a, c):
    """Fail before model creation if A/C differ in matched training conditions."""
    for section, keys in {
        'training': ['seed','lora_rank','lora_alpha','batch_size','gradient_accumulation_steps',
                     'max_steps','learning_rate','weight_decay','scheduler_type','warmup_steps','grad_clip_norm'],
        'dataset': ['conditioning','episode_manifest'],
        'finetune': ['checkpoint_path'],
        'execution': ['control_dt_s','clock_provenance'],
    }.items():
        for key in keys:
            if a[section].get(key) != c[section].get(key):
                raise ValueError(f'unmatched A/C configuration: {section}.{key}')
    if a['model']['inference'] != c['model']['inference']:
        raise ValueError('unmatched A/C inference settings')


def rollout(env, policy, deploy, *, synchronize=lambda: None):
    """Reuse deployed chunk execution; classify failures without skipping scenes."""
    calls, timings, simulated = 0, [], 0.
    simulated_known = True
    try:
        while env.take_action_cnt < env.step_lim and not env.eval_success:
            observation = env.get_obs()
            synchronize()
            start = time.perf_counter()
            calls += 1
            policy.last_controller_result = None
            try:
                deploy.eval(env, policy, observation)
            finally:
                synchronize()
                timings.append(time.perf_counter()-start)
                prediction_log = getattr(policy, 'prediction_log', None)
                if prediction_log is not None:
                    parameters = {key:getattr(policy,key,None) for key in
                        ('last_predicted_parameters','last_raw_parameters','last_postprocessed_parameters')}
                    prediction_log.write(json.dumps(dict(call=calls,**parameters),
                        default=lambda x:x.tolist())+'\n')
                    prediction_log.flush()
                controller = policy.last_controller_result
                if controller is not None and controller.get('simulated_duration_s') is not None:
                    simulated += controller['simulated_duration_s']
                elif controller is not None:
                    simulated_known = False
            controller = policy.last_controller_result
            if controller is None or controller['action_commands'] <= 0:
                raise ValueError('controller made no progress')
        status = 'success' if env.eval_success else 'timeout'
        reason = None if env.eval_success else 'controller step limit reached'
    except TimeoutError as error:
        status, reason = 'timeout', str(error)
    except Exception as error:
        status = 'invalid_prediction' if policy.prediction_counts.get('invalid_predictions', 0) else 'rollout_error'
        reason = f'{type(error).__name__}: {error}'
    return dict(status=status, reason=reason, controller_steps=env.take_action_cnt,
                policy_calls=calls, simulated_duration_s=simulated if simulated_known else None,
                observation_to_chunk_execution_wall_s=timings,
                timing_scope='input processing + inference + controller; see policy_timings for inference latency',
                policy_timings=getattr(policy, 'call_timings', []),
                counts=policy.prediction_counts.copy())


def same_scene(actual, expected):
    import numpy as np
    return actual.keys() == expected.keys() and all(
        np.shape(value) == np.shape(expected[key]) and
        np.allclose(value, expected[key], atol=1e-6, rtol=0) for key, value in actual.items())


def evaluate(args, plan, cases):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('learned-policy testing requires a Slurm GPU allocation')
    import numpy as np
    import torch
    import yaml
    if not torch.cuda.is_available():
        raise RuntimeError('no allocated CUDA device')
    for path in (REFERENCE, ROOT):
        sys.path.insert(0, str(path))
    from inference.robotwin.Motus import deploy_policy as deploy
    from trajectory_tools.replay import load_config, scene_snapshot
    args.output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(args.config.read_text())
    initial = args.original_checkpoint or Path(config['finetune']['checkpoint_path']).resolve()
    if 'A' in args.policies and 'C' in args.policies:
        validate_matched_configs(yaml.safe_load((args.dense_checkpoint/'resolved_config.yaml').read_text()),
                                 yaml.safe_load((args.spline_checkpoint/'resolved_config.yaml').read_text()))
    (args.output/'test_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    initial_weights = initial/'mp_rank_00_model_states.pt' if initial.is_dir() else initial
    summary = dict(status='incomplete', comparison='original_and_matched_dense_spline_lora',
                   attribution='A versus C combines action representation and video alignment; original is separately pretrained',
                   dataset_manifest=str(args.manifest.resolve()) if args.manifest else None, dataset_identity=plan.get('dataset_identity'),
                   test_plan_sha256=sha256(args.plan), demonstration_seeds_sha256=sha256(args.demo_seeds) if args.demo_seeds else None,
                   pretrained_checkpoint=str(initial),
                   pretrained_weights_sha256=sha256(initial_weights),
                   task=plan['task_name'], task_config=plan['task_config'], policies={})
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    baseline = {}
    previous = Path.cwd()
    try:
        os.chdir(REFERENCE)  # Reused simulator asset loading requires its repo root.
        task_type = getattr(importlib.import_module(f"envs.{plan['task_name']}"), plan['task_name'])
        task_config = load_config(plan['task_name'], plan['task_config'])
        task_config['eval_video_log'] = False
        task_config['data_type'].update(rgb=True, qpos=True, endpose=True)
        for kind in args.policies:
            directory = args.output/kind
            directory.mkdir()
            checkpoint = initial if kind == 'original' else getattr(args, 'dense_checkpoint' if kind == 'A' else 'spline_checkpoint')
            resolved = copy.deepcopy(config) if kind == 'original' else yaml.safe_load((checkpoint/'resolved_config.yaml').read_text())
            resolved['common']['action_representation'] = 'spline' if kind == 'C' else 'dense'
            if kind == 'original':
                resolved['training']['lora'] = False
            else:
                if resolved['dataset']['episode_manifest'] != str(args.manifest.resolve()):
                    raise ValueError('checkpoint manifest differs from frozen evaluation manifest')
                if resolved['execution']['control_dt_s'] != plan['control_dt_s'] or resolved['execution']['clock_provenance'] != plan['clock_provenance']:
                    raise ValueError('checkpoint controller clock differs from evaluation plan')
            resolved['evaluation'] = dict(checkpoint_kind='pretrained' if kind == 'original' else 'native_lora',
                                          checkpoint_report=str(directory/'checkpoint_load.json'))
            config_path = directory/'resolved_config.yaml'
            config_path.write_text(yaml.safe_dump(resolved, sort_keys=False))
            policy = deploy.MotusPolicy(str(checkpoint),
                str(config_path), resolved['model']['wan']['config_path'], resolved['model']['vlm']['checkpoint_path'],
                device='cuda', log_dir=str(directory), task_name=plan['task_name'])
            torch.cuda.reset_peak_memory_stats()
            records = []
            for index, case in enumerate(cases):
                start = time.perf_counter()
                env = task_type()
                trace = None
                telemetry = (directory/f"scene-{case['seed']}-physics.jsonl").open('x')
                policy.prediction_log = (directory/f"scene-{case['seed']}-predictions.jsonl").open('x')
                row = dict(seed=case['seed'], instruction=case['instruction'], status='reset_error',
                           policy_seed=int(case.get('policy_seed', case['seed'])), training_seed=config['training']['seed'])
                try:
                    random.seed(case['seed']); np.random.seed(case['seed']); torch.manual_seed(case['seed'])
                    env.setup_demo(now_ep_num=index, seed=case['seed'], is_test=True, **copy.deepcopy(task_config))
                    row['initial_scene'] = scene_snapshot(env)
                    if case['seed'] not in baseline and kind == args.policies[0]:
                        baseline[case['seed']] = row['initial_scene']
                    elif case['seed'] not in baseline or not same_scene(row['initial_scene'], baseline[case['seed']]):
                        row['status'] = 'scene_mismatch_error'
                        raise ValueError('scene physics do not match the original policy reset')
                    if env.check_success():
                        raise ValueError('scene already successful before policy execution')
                    trace = SceneTrace(env.scene, env.robot, telemetry)
                    env.scene = trace
                    env.set_instruction(instruction=case['instruction'])
                    deploy.reset_model(policy)
                    policy.prediction_counts = dict(invalid_predictions=0, short_coverage=0, repairs=0,
                                                    fallbacks=0, knot_repairs=0, gripper_saturations=0)
                    policy.call_timings = []
                    torch.manual_seed(row['policy_seed']); torch.cuda.manual_seed_all(row['policy_seed'])
                    row.update(rollout(env, policy, deploy, synchronize=torch.cuda.synchronize))
                except Exception as error:
                    row.update(reason=f'{type(error).__name__}: {error}')
                finally:
                    telemetry.close()
                    policy.prediction_log.close()
                    if trace is not None:
                        row['physics_steps'] = trace.steps
                        row['simulated_duration_s'] = trace.elapsed
                        row['telemetry'] = telemetry.name
                        env.scene = trace.scene
                    if hasattr(env, 'scene'):
                        try:
                            env.close_env(clear_cache=True)
                        except Exception as error:
                            row['status'] = 'cleanup_error'
                            row['cleanup_error'] = f'{type(error).__name__}: {error}'
                row['wall_time_s'] = time.perf_counter()-start
                row['checkpoint'] = str(checkpoint)
                row['resolved_config'] = str(config_path)
                row['peak_allocated_gpu_bytes'] = torch.cuda.max_memory_allocated()
                records.append(row)
                (directory/'episodes.json').write_text(json.dumps(records, indent=2)+'\n')
                summary['policies'][kind] = summarize(records, len(cases))
                (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
                print(kind, case['seed'], row['status'], flush=True)
            del policy
            gc.collect(); torch.cuda.empty_cache()
    finally:
        os.chdir(previous)
    if 'A' in args.policies and 'C' in args.policies:
        a_rows = json.loads((args.output/'A/episodes.json').read_text())
        c_rows = json.loads((args.output/'C/episodes.json').read_text())
        joint = [(a,c) for a,c in zip(a_rows,c_rows) if a['seed']==c['seed'] and a['status']==c['status']=='success']
        summary['jointly_successful_scenes'] = len(joint)
        summary['paired_success_wall_times_s'] = [dict(seed=a['seed'],A=a['wall_time_s'],C=c['wall_time_s']) for a,c in joint]
    summary['status'] = 'completed'
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)
    # Policy failures are measured outcomes; infrastructure errors are also visible
    # as a failed job, while their scenes stay in the reported denominator.
    if any(any(key.endswith('error') for key in report['outcomes']) for report in summary['policies'].values()):
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/robotwin_ac.yaml')
    parser.add_argument('--policies', nargs='+', choices=['original','A','C'], default=['original'])
    parser.add_argument('--original-checkpoint', type=Path)
    parser.add_argument('--dense-checkpoint', type=Path)
    parser.add_argument('--demo-seeds', type=Path, help='authoritative seed.txt for the selected demonstrations')
    parser.add_argument('--plan', required=True, type=Path, help='frozen held-out scene seeds, instructions and verified controller clock')
    parser.add_argument('--spline-checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for key in ('manifest', 'demo_seeds', 'plan', 'spline_checkpoint', 'dense_checkpoint', 'original_checkpoint', 'config', 'output'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    plan = json.loads(args.plan.read_text())
    if len(set(args.policies)) != len(args.policies):
        parser.error('duplicate policy selection')
    task_config = plan.get('task_config')
    if task_config not in ('demo_clean', 'demo_randomized'):
        raise ValueError('unsupported task configuration')
    if plan.get('task_config_sha256') != sha256(REFERENCE/f'env_cfg/task_config/{task_config}.yml'):
        raise ValueError('plan must be bound to the installed simulator task configuration')
    if not (REFERENCE/f"envs/{plan['task_name']}.py").is_file():
        raise ValueError('task identifier is not installed')
    if args.policies == ['original']:
        # Original evaluation does not depend on conversion/replay or a trained C.
        cases = plan.get('cases', [])
        if not cases or len({c['seed'] for c in cases}) != len(cases):
            raise ValueError('declare distinct scenes before original evaluation')
        for case in cases:
            if type(case['seed']) is not int or case['seed'] < 0 or not case.get('instruction', '').strip():
                raise ValueError('each scene needs a nonnegative seed and fixed instruction')
    else:
        if args.manifest is None or args.demo_seeds is None:
            parser.error('A/C require --manifest and authoritative --demo-seeds')
        manifest = json.loads(args.manifest.read_text())
        identity = {k:v for k,v in manifest.items() if k != 'identity'}
        if manifest.get('identity') != hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest():
            raise ValueError('dataset manifest identity is invalid')
        seeds = [int(value) for value in args.demo_seeds.read_text().split()]
        if plan.get('demonstration_seeds_sha256') != sha256(args.demo_seeds):
            raise ValueError('plan is not bound to the authoritative demonstration seed file')
        cases = validate_plan(plan, manifest, seeds)
        for kind, checkpoint in [('A',args.dense_checkpoint), ('C',args.spline_checkpoint)]:
            if kind in args.policies and checkpoint is None and not args.check:
                parser.error(f'{kind} requires its checkpoint')
    print(f"Plan checked: {len(cases)} fixed scenes; policies={args.policies}", flush=True)
    if args.check:
        return
    if args.output is None:
        parser.error('--output required for testing')
    evaluate(args, plan, cases)


if __name__ == '__main__':
    main()
