"""Stage localization through MotusPolicy's deployed decoder; no training."""
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motus_actions.spline import AffineScale, local_curve


def decoder_policy(manifest,cpu=False):
    if cpu:
        # Lightweight login-node numerical check when Torch is unavailable.
        # Execute the exact deployed method source, with no copied decoder.
        # This does NOT pass the full dataset/model entry-point gates.
        import ast
        from types import SimpleNamespace,MethodType
        path=ROOT/'inference/robotwin/Motus/deploy_policy.py'
        cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MotusPolicy')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='decode_spline_prediction')
        module=ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[]))
        namespace={'np':np}
        exec(compile(module,str(path),'exec'),namespace)
        policy=SimpleNamespace()
        policy.decode_spline_prediction=MethodType(namespace['decode_spline_prediction'],policy)
    else:
        from inference.robotwin.Motus.deploy_policy import MotusPolicy
        policy = MotusPolicy.__new__(MotusPolicy)
    policy.config_dict = {'common': {'global_downsample_rate': 3}}
    policy.spline_parameter_scale = AffineScale(manifest['parameter_low'], manifest['parameter_high'])
    policy.spline_action_scale = AffineScale(manifest['action_low'], manifest['action_high'])
    policy.prediction_counts = dict(knot_repairs=0, repairs=0)
    return policy


def knot_record(parameters):
    if parameters is None:
        return dict(valid=False,reason='normalization rejected before local knots were available')
    knots = np.asarray(parameters)[:, 0]
    values, counts = np.unique(knots, return_counts=True)
    record = dict(knots=knots.tolist(), spacing=np.diff(knots).tolist(),
        domain=[float(knots[3]), float(knots[-4])],
        multiplicities=[dict(time=float(v), count=int(c)) for v,c in zip(values,counts)],
        time_units='source_sample_index', origin='current_observation', effective_control_rows=[0,31])
    record['domain_covers_matched_prefix']=bool(knots[3]<=3 and knots[-4]>=48)
    try:
        curve = local_curve(parameters)
        record.update(valid=True, trimmed_knots=len(curve.t), domain=[float(curve.t[3]),float(curve.t[-4])])
    except ValueError as error:
        record.update(valid=False, reason=str(error))
    return record


def trajectory_metrics(commands, truth, scale, offsets):
    residual = scale.encode(commands) - scale.encode(truth)
    def events(values):
        answer = []
        for channel in (6,13):
            x = values[:, channel]
            indices = np.flatnonzero((x[:-1] >= .5) != (x[1:] >= .5))
            answer.append([(int(np.sign(x[i+1]-x[i])), float(offsets[i] +
                (.5-x[i])/(x[i+1]-x[i])*(offsets[i+1]-offsets[i]))) for i in indices])
        return answer
    expected, actual = events(truth), events(commands)
    aligned = all([e[0] for e in a] == [e[0] for e in b] for a,b in zip(expected,actual))
    errors = [abs(a[1]-b[1]) for arm_a,arm_b in zip(expected,actual) for a,b in zip(arm_a,arm_b)]
    return dict(trajectory_normalized_rmse=float(np.sqrt(np.mean(residual**2))),
        trajectory_normalized_max=float(np.abs(residual).max()),
        endpoint_normalized_max=float(np.abs(residual[-1]).max()),
        gripper_events_expected=expected, gripper_events_predicted=actual,
        gripper_event_count_direction_match=aligned,
        gripper_transition_max_error_samples=max(errors,default=0.) if aligned else None)


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--all-windows',action='store_true')
    args=parser.parse_args()
    output = args.output or ROOT/'outputs'/f'decode-stages-{os.environ.get("SLURM_JOB_ID", "cpu")}'
    output.mkdir(exist_ok=False)
    import yaml
    audited=yaml.safe_load((ROOT/'outputs/tiny-C-7471647/checkpoint_step_50/resolved_config.yaml').read_text())
    source = Path(audited['dataset']['episode_manifest'])
    manifest = json.loads(source.read_text())
    policy = decoder_policy(manifest,cpu=args.cpu)
    entry = next(e for e in manifest['episodes'] if e['id'] in manifest['split']['train'])
    with np.load(entry['cache_path']) as cached:
        slot = len(cached['observations'])//2
        current = int(cached['observations'][slot])
        local = cached['windows'][slot].copy()
        global_curve = BSpline(cached['knots'], cached['controls'],3,extrapolate=False)
        expected = policy.spline_action_scale.decode(global_curve(current+np.arange(1,17)*3))
    packed = policy.spline_parameter_scale.encode(local)
    stages = [('a_ground_truth_local',local,False),
              ('b_normalization_roundtrip',packed,True)]
    if not args.cpu:
        import torch
        from data.dataset import collate_fn
        sample = dict(first_frame=torch.zeros(3,4,4),video_frames=torch.zeros(8,3,4,4),
            initial_state=torch.zeros(14),action_sequence=torch.tensor(packed,dtype=torch.float32),
            language_embedding=None,vlm_inputs=None)
        formatted = collate_fn([sample])['action_sequence'].to(torch.bfloat16)[0].float().numpy()
        stages.append(('c_model_target_bfloat16',formatted,True))
    results=[]
    for name,parameters,normalized in stages:
        raw = policy.spline_parameter_scale.decode(parameters) if normalized else parameters
        record = dict(stage=name,knots=knot_record(raw),repairs=0)
        try:
            commands = policy.decode_spline_prediction(parameters, normalized=normalized, predicted=False)
            record.update(status='passed', **trajectory_metrics(commands,expected,policy.spline_action_scale,np.arange(1,17)*3))
        except ValueError as error:
            record.update(status='failed',reason=str(error))
        results.append(record)
    if args.cpu:
        for stage in ('c_model_target_bfloat16','d_actual_euler_oracle'):
            results.append(dict(stage=stage,status='blocked',reason='Torch available only in allocated container; diagnostics budget pending'))
    else:
        from test_interfaces import oracle_inference
        inferred = oracle_inference(torch.tensor(packed,dtype=torch.float32)[None])[0].numpy()
        commands = policy.decode_spline_prediction(inferred,predicted=False)
        results.append(dict(stage='d_actual_euler_oracle',status='passed',
            target='noise-clean; x(sigma)=(1-sigma)*clean+sigma*noise; integrate sigma 1->0',
            packed_max_error=float(np.abs(inferred-packed).max()),
            **trajectory_metrics(commands,expected,policy.spline_action_scale,np.arange(1,17)*3)))
    prediction_path=ROOT/'outputs/wrapper-inference-7471966/C-prediction.npz'
    with np.load(prediction_path) as data:
        learned=data['packed'].copy()
    record=dict(stage='e_legacy_learned_prediction',conditioning='raw_instruction_train/prefixed_deploy_INCOMPATIBLE',
                source=str(prediction_path),raw=knot_record(policy.spline_parameter_scale.decode(learned)))
    try:
        policy.decode_spline_prediction(learned)
        record['status']='passed'
    except ValueError as error:
        record.update(status='failed',reason=str(error))
    record.update(postprocessed=knot_record(policy.last_postprocessed_parameters),repairs=policy.prediction_counts['knot_repairs'])
    arrays=dict(packed=policy.last_predicted_parameters)
    if policy.last_raw_parameters is not None: arrays['raw']=policy.last_raw_parameters
    if policy.last_postprocessed_parameters is not None: arrays['postprocessed']=policy.last_postprocessed_parameters
    np.savez(output/'learned_knots.npz',**arrays)
    results.append(record)
    report=dict(mode='exact_deployed_method_source_CPU' if args.cpu else 'full_deployment_module',
        source_manifest=str(source),episode=entry['id'],observation=current,controller_offsets=(np.arange(1,17)*3).tolist(),
        stages=results,first_failed_stage=next((r['stage'] for r in results if r['status']=='failed'),None),
        classification_complete=all(r['status']=='passed' for r in results[:4]),
        blocked_stages=[r['stage'] for r in results if r['status']=='blocked'])
    if args.all_windows:
        worst=0.; count=0; horizons=[]
        for entry in manifest['episodes']:
            with np.load(entry['cache_path']) as cached:
                global_curve=BSpline(cached['knots'],cached['controls'],3,extrapolate=False)
                for current,local in zip(cached['observations'],cached['windows']):
                    expected=policy.spline_action_scale.decode(global_curve(current+np.arange(1,17)*3))
                    for parameters,normalized in [(local,False),(policy.spline_parameter_scale.encode(local),True)]:
                        commands=policy.decode_spline_prediction(parameters,normalized=normalized,predicted=False)
                        worst=max(worst,float(np.abs(policy.spline_action_scale.encode(commands)-policy.spline_action_scale.encode(expected)).max()))
                    horizons.append(local[[3,32],0].tolist())
                    count+=1
        values=np.asarray(horizons)
        report['all_windows']=dict(count=count,local_global_normalized_max_error=worst,repairs=0,
            valid_left_source_samples_range=[float(values[:,0].min()),float(values[:,0].max())],
            valid_right_source_samples_range=[float(values[:,1].min()),float(values[:,1].max())],
            support_duration_source_samples_quantiles=np.quantile(values[:,1]-values[:,0],[0,.5,1]).tolist(),
            physical_duration_seconds='blocked: authoritative source clock unavailable')
        if worst>1e-5: raise ValueError(f'local/global gate failed: {worst}')
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='stages'}),flush=True)
    if any(r['status']!='passed' for r in results[:4]):
        raise SystemExit(2)


if __name__=='__main__':
    main()
