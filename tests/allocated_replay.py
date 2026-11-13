"""Original-first replay through the existing RoboTwin fixed-rate controller."""
import argparse
import copy
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'WAM-RoboTwin'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(REFERENCE))
from motus_actions.spline import AffineScale,decode,extract_local,controller_commands
from motus_actions.prepare import digest
# The replay module is standalone. Load that verified file without importing
# trajectory_tools' unrelated HDF5/fitting package initializers on CPU checks.
spec=importlib.util.spec_from_file_location('robotwin_replay',REFERENCE/'trajectory_tools/replay.py')
native_replay=importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_replay)
execute_fixed_rate=native_replay.execute_fixed_rate
load_config=native_replay.load_config
scene_snapshot=native_replay.scene_snapshot


def replay_plans(cached,scale,source_dt):
    actions=cached['actions']
    curve=BSpline(cached['knots'],cached['controls'],3,extrapolate=False)
    indices=np.arange(3,len(actions),3)
    reconstructed=[]
    for start in range(0,len(indices),16):
        current=0 if start==0 else int(indices[start-1])
        reconstructed.append(scale.decode(decode(extract_local(curve,current),indices[start:start+16]-current)))
    spline=np.concatenate(reconstructed)
    original=actions[indices]
    boundaries=list(range(16,len(indices),16))
    normalized=scale.encode(spline)-scale.encode(original)
    extra=[float(np.abs((scale.encode(spline[i])-scale.encode(spline[i-1]))-
        (scale.encode(original[i])-scale.encode(original[i-1]))).max()) for i in boundaries]
    metrics=dict(chunk_boundaries=boundaries,source_completion_indices=indices.tolist(),
        max_normalized_reconstruction_error=float(np.abs(normalized).max()),
        additional_transition_jump_max=max(extra,default=0.),executed_prefix_rows=16,
        final_prefix_rows=len(indices)%16 or 16)
    expected_events=gripper_events(np.vstack([actions[0],original]),np.r_[0,indices]*source_dt)
    reconstructed_events=gripper_events(np.vstack([actions[0],spline]),np.r_[0,indices]*source_dt)
    match=all([e['direction'] for e in a]==[e['direction'] for e in b]
        for a,b in zip(expected_events,reconstructed_events))
    shifts=[abs(a['time_s']-b['time_s'])/source_dt for x,y in zip(expected_events,reconstructed_events) for a,b in zip(x,y)]
    metrics.update(gripper_event_count_direction_match=match,
        gripper_transition_max_error_source_samples=max(shifts,default=0.) if match else None)
    def plan(values,dt):
        return SimpleNamespace(actions=values,control_dt=dt,
            execution_times=np.arange(len(values))*dt,requested_duration=len(values)*dt)
    return [('original_full',plan(actions[1:],source_dt)),
        ('original_matched',plan(original,3*source_dt)),('spline_matched',plan(spline,3*source_dt))],metrics


def gripper_events(states,times):
    result=[]
    for channel in (6,13):
        x=np.asarray(states)[:,channel]
        crossings=np.flatnonzero((x[:-1]>=.5)!=(x[1:]>=.5))+1
        result.append([dict(time_s=float(times[i-1]+(.5-x[i-1])/(x[i]-x[i-1])*(times[i]-times[i-1])),
            direction=int(np.sign(x[i]-x[i-1]))) for i in crossings])
    return result


def measured_state(robot):
    measured=robot.get_left_arm_real_jointState()+robot.get_right_arm_real_jointState()
    # The native getter returns commanded gripper values, even in its
    # "real" variant. Invert the base-finger qpos mapping independently.
    for side,column in [('left',6),('right',13)]:
        entity=getattr(robot,f'{side}_entity')
        joint,multiplier,bias=getattr(robot,f'{side}_gripper')[0]
        qpos=entity.get_qpos()[entity.get_active_joints().index(joint)]
        low,high=getattr(robot,f'{side}_gripper_scale')
        measured[column]=float(((qpos-bias)/multiplier-low)/(high-low))
    return measured


class MeasuredScene:
    def __init__(self,scene,robot):
        self.scene,self.robot,self.states=scene,robot,[measured_state(robot)]
    def get_timestep(self): return self.scene.get_timestep()
    def step(self):
        self.scene.step()
        self.states.append(measured_state(self.robot))


def execute_measured(env,plan,output,tracking_tolerance_rad=.1):
    limits=[np.asarray(j.get_limits())[0] for side in ('left','right')
        for j in getattr(env.robot,f'{side}_arm_joints')]
    commands,saturations=controller_commands(plan.actions,limits)
    plan=copy.copy(plan); plan.actions=commands
    measured=MeasuredScene(env.scene,env.robot)
    proxy=SimpleNamespace(scene=measured,robot=env.robot,check_success=env.check_success)
    start=time.perf_counter()
    result=execute_fixed_rate(proxy,plan,tracking_tolerance_rad=tracking_tolerance_rad)
    result.update(wall_time_s=time.perf_counter()-start,gripper_saturations=saturations,
        repairs=saturations,fallbacks=0,controller_protocol='native_joint_drives_fixed_rate',early_success_exit=False)
    states=np.asarray(measured.states,dtype=float)
    times=np.arange(len(states))*env.scene.get_timestep()
    result['commanded_gripper_events']=gripper_events(np.vstack([states[0],commands]),np.arange(len(commands)+1)*plan.control_dt)
    result['measured_gripper_events']=gripper_events(states,times) if len(states) else [[],[]]
    np.savez(output,measured_states=states,physics_times_s=times,commands=commands,
        command_completion_times_s=np.arange(1,len(commands)+1)*plan.control_dt)
    return result


def compare_scene(actual,expected):
    if actual.keys()!=expected.keys() or any(np.shape(value)!=np.shape(expected[key]) or
        not np.allclose(value,expected[key],rtol=0,atol=1e-6) for key,value in actual.items()):
        raise ValueError('initial physics state differs from recorded demonstration')


def run_episode(entry,cached,metadata,output,tracking_tolerance_rad=None):
    config=load_config('place_can_basket','demo_clean')
    if metadata['demonstration_sha256']!=entry['sha256'] or metadata['task_config_sha256']!=digest(REFERENCE/'env_cfg/task_config/demo_clean.yml'):
        raise ValueError('scene sidecar dataset/config identity mismatch')
    if metadata['controller_protocol']!='native_drives_target_next_sample_v1':
        raise ValueError('recorded action timing/drive semantics not verified for this controller')
    source_dt=float(metadata['source_dt_s'])
    if not np.isfinite(source_dt) or source_dt<=0 or not metadata['source_dt_provenance']:
        raise ValueError('missing verified source clock')
    if not metadata['initial_scene'] or not metadata['object_identity']:
        raise ValueError('seed alone is insufficient: full initial scene and object identities required')
    # Construct spline chunks only after both original paths execute. A local
    # coverage failure must never prevent the original replay diagnostic.
    actions=cached['actions']
    def original_plan(values,dt):
        return SimpleNamespace(actions=values,control_dt=dt,
            execution_times=np.arange(len(values))*dt,requested_duration=len(values)*dt)
    plans=[('original_full',original_plan(actions[1:],source_dt)),
           ('original_matched',original_plan(actions[3::3],3*source_dt))]
    metrics=None
    results=[]
    previous=Path.cwd()
    try:
        os.chdir(REFERENCE)  # Existing asset paths require the RoboTwin root.
        task_type=getattr(importlib.import_module('envs.place_can_basket'),'place_can_basket')
        for name,plan in plans:
            env=task_type()
            try:
                random.seed(metadata['seed'])
                env.setup_demo(now_ep_num=metadata['episode_index'],seed=metadata['seed'],is_test=True,**copy.deepcopy(config))
                identity=dict(can_name=env.can_name,can_id=int(env.can_id),basket_name=env.basket_name,
                    basket_id=int(env.basket_id),arm_tag=str(env.arm_tag))
                if identity!=metadata['object_identity']: raise ValueError('recorded object identity mismatch')
                compare_scene(scene_snapshot(env),metadata['initial_scene'])
                initial=np.asarray(env.robot.get_left_arm_real_jointState()+env.robot.get_right_arm_real_jointState())
                if np.max(np.abs(initial-cached['states'][0]))>1e-3:
                    raise ValueError('initial measured robot state differs from demonstration')
                if env.check_success(): raise ValueError('task already successful before replay')
                result=execute_measured(env,plan,output/f'{name}.npz',
                    tracking_tolerance_rad=tracking_tolerance_rad if tracking_tolerance_rad is not None else .1)
                result.update(variant=name,exact_initial_physics_checked=True,source_dt_s=source_dt)
                results.append(result)
                (output/'execution.json').write_text(json.dumps(results,indent=2)+'\n')
                if result['status']!='completed' or not result['success'] or (
                    tracking_tolerance_rad is not None and result['tracking_failure']):
                    break  # Diagnose original failures before any spline attribution.
                if name=='original_matched':
                    try:
                        reconstructed,metrics=replay_plans(cached,AffineScale(cached['action_low'],cached['action_high']),source_dt)
                        plans.append(reconstructed[-1])
                    except ValueError as error:
                        results.append(dict(variant='spline_matched',status='decoder_failure',reason=str(error)))
                        (output/'execution.json').write_text(json.dumps(results,indent=2)+'\n')
                        break
            finally:
                if hasattr(env,'scene'): env.close_env(clear_cache=True)
    finally:
        os.chdir(previous)
    return dict(offline_metrics=metrics,executions=results)


def assess_replay(episodes,tracking_tolerance_rad):
    gates=dict(original_replay='blocked',spline_replay='blocked',matched_controller='blocked')
    original=[]; spline=[]; metrics=[]
    for episode in episodes:
        for result in episode.get('executions',[]):
            (spline if result['variant']=='spline_matched' else original).append(result)
        if episode.get('offline_metrics') is not None: metrics.append(episode['offline_metrics'])
    def failed(rows):
        return any(r['status']!='completed' or not r.get('success') or
            (tracking_tolerance_rad is not None and r['tracking_max_abs_rad']>tracking_tolerance_rad) for r in rows)
    if failed(original): gates['original_replay']='failed'
    if failed(spline): gates['spline_replay']='failed'
    complete=len(episodes)>=5 and all(e['status']=='executed' for e in episodes)
    if complete and len(original)==2*len(episodes):
        gates['matched_controller']='passed'
        if tracking_tolerance_rad is not None and not failed(original): gates['original_replay']='passed'
    if len(spline)==len(episodes) and len(metrics)==len(episodes):
        numerical=all(m['max_normalized_reconstruction_error']<=.01 and m['additional_transition_jump_max']<=.01
            and m['gripper_event_count_direction_match'] and m['gripper_transition_max_error_source_samples']<=1 for m in metrics)
        if not numerical: gates['spline_replay']='failed'
        elif gates['original_replay']=='passed' and not failed(spline): gates['spline_replay']='passed'
    return gates


def headless_render():
    """Retain the synthetic rendering diagnostic, separate from replay."""
    import sapien.core as sapien
    engine=sapien.Engine(); engine.set_renderer(sapien.SapienRenderer())
    scene=engine.create_scene(); scene.set_ambient_light([.5,.5,.5]); scene.add_ground(0)
    camera=scene.add_camera('smoke',64,64,1.,.1,10.)
    camera.set_pose(sapien.Pose([0,0,1])); scene.update_render(); camera.take_picture()
    color=camera.get_picture('Color')
    if color.shape!=(64,64,4) or not np.isfinite(color).all(): raise ValueError('invalid headless render')
    return dict(headless='passed',render_shape=list(color.shape),original_replay='not_run',spline_replay='not_run')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scene-metadata',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--episode-manifest',type=Path,default=ROOT/'outputs/pilot-cache-guard-20261111/manifest.json')
    parser.add_argument('--tracking-tolerance-rad',type=float,
        help='must be agreed and recorded before replay acceptance; no default acceptance threshold')
    parser.add_argument('--headless-only',action='store_true')
    args=parser.parse_args()
    output=args.output or ROOT/'outputs'/f'replay-repair-{os.environ.get("SLURM_JOB_ID","preflight")}'
    output.mkdir(parents=True,exist_ok=False)
    if args.tracking_tolerance_rad is not None and (not np.isfinite(args.tracking_tolerance_rad) or args.tracking_tolerance_rad<=0):
        parser.error('tracking tolerance must be finite and positive')
    if args.headless_only:
        if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('headless rendering requires a Slurm GPU allocation')
        report=headless_render()
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        return
    manifest=json.loads(args.episode_manifest.read_text())
    entries=[e for e in manifest['episodes'] if e['id'] in manifest['split']['validation']][:5]
    sidecar=json.loads(args.scene_metadata.read_text()) if args.scene_metadata else {}
    report=dict(original_replay='blocked',spline_replay='blocked',matched_controller='blocked',episodes=[],
        tracking_acceptance_rad=args.tracking_tolerance_rad,
        required_input='recorded initial physics/object identities + verified source clock/action timing, bound to each raw demonstration hash; predeclared tracking tolerance for acceptance')
    for entry in entries:
        row=dict(episode=entry['id'],demonstration_sha256=entry['sha256'])
        if entry['id'] not in sidecar:
            row.update(status='blocked',reason='missing authoritative scene/timing sidecar; no commands executed')
        else:
            if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('simulator replay requires a Slurm allocation')
            target=output/entry['episode_name']; target.mkdir()
            with np.load(entry['cache_path']) as data: cached={key:data[key] for key in data.files}
            cached.update(action_low=manifest['action_low'],action_high=manifest['action_high'])
            try: row.update(status='executed',**run_episode(entry,cached,sidecar[entry['id']],target,args.tracking_tolerance_rad))
            except (KeyError,ValueError) as error: row.update(status='blocked',reason=str(error))
        report['episodes'].append(row)
        report.update(assess_replay(report['episodes'],args.tracking_tolerance_rad))
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if any(report[key]!='passed' for key in ('original_replay','spline_replay','matched_controller')):
        raise SystemExit(2)


if __name__=='__main__': main()
