"""Score held-out RoboTwin commands in joint and end-effector space."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .rollout_policy import ROBOTWIN_ROOT, load_checkpoint
from .simulator import robotwin_api, write_json

ARM_CHANNELS = [*range(6), *range(7, 13)]


class RoboTwinFK:
    """Use the simulator's articulation and existing world-frame TCP getters."""

    def __init__(self, task_args):
        import sapien.core as sapien
        from envs.robot.robot import Robot

        if task_args['embodiment'] != ['aloha-agilex']:
            raise ValueError('Expected training embodiment aloha-agilex')
        self.engine = sapien.Engine()
        self.scene = self.engine.create_scene()
        self.robot = Robot(self.scene, **task_args)
        self.robot.init_joints()
        self.entity = self.robot.left_entity
        active = [joint.get_name() for joint in self.entity.get_active_joints()]
        names = self.robot.left_arm_joints_name + self.robot.right_arm_joints_name
        self.arm_indices = [active.index(name) for name in names]
        self.qpos = self.entity.get_qpos()
        if len(names) != 12 or len(set(self.arm_indices)) != 12:
            raise ValueError('Expected two six-joint arms')

    def poses(self, commands):
        poses = []
        for command in commands:
            qpos = self.qpos.copy()
            qpos[self.arm_indices] = command[ARM_CHANNELS]
            # Gripper coordinates are downstream of both TCP reference joints.
            self.entity.set_qpos(qpos)
            poses.append([self.robot.get_left_tcp_pose(), self.robot.get_right_tcp_pose()])
        return np.asarray(poses)

    def check(self):
        """Check articulation TCPs against SAPIEN's Pinocchio FK at two poses."""
        home = np.zeros(14)
        home[:6], home[7:13] = self.robot.left_homestate, self.robot.right_homestate
        perturbed = home.copy()
        perturbed[ARM_CHANNELS] += np.linspace(-.25, .25, 12)
        pinocchio = self.entity.create_pinocchio_model()
        links = [link.get_name() for link in self.entity.get_links()]
        max_position = max_orientation = 0.
        for command in (home, perturbed):
            actual = self.poses([command])[0]
            pinocchio.compute_forward_kinematics(self.entity.get_qpos())
            for i, arm in enumerate(('left', 'right')):
                joint = getattr(self.robot, f'{arm}_ee')
                link_id = links.index(joint.child_link.get_name())
                pose = self.entity.get_root_pose() * pinocchio.get_link_pose(link_id) * joint.get_pose_in_child()
                configured = (pose.to_transformation_matrix()[:3, :3]
                              @ getattr(self.robot, f'{arm}_global_trans_matrix')
                              @ getattr(self.robot, f'{arm}_delta_matrix'))
                tcp = pose.p + configured[:, 0] * getattr(self.robot, f'{arm}_gripper_bias')
                max_position = max(max_position, float(np.linalg.norm(tcp - actual[i, :3])))
                max_orientation = max(max_orientation, float(
                    (Rotation.from_matrix(configured).inv() * rotations(actual[i])).magnitude().item()))
        if max_position > 1e-5 or max_orientation > 1e-5:
            raise ValueError(f'RoboTwin TCP and Pinocchio FK disagree: {max_position} m, {max_orientation} rad')
        return dict(max_position_m=max_position, max_orientation_rad=max_orientation)


def rotations(poses):
    return Rotation.from_quat(poses[..., [4, 5, 6, 3]].reshape(-1, 4))


def score_prediction(parameters, times, observation, contract):
    """Decode only in-domain timestamps through the existing strict decoder."""
    from data.robotwin2.adaptive_spline import prepare_local, evaluate_local

    valid = np.zeros(len(times), dtype=bool)
    diagnostics = {}
    if contract['representation'] == 'dense':
        if parameters.shape != (len(times), 14):
            raise ValueError(f'Unexpected dense prediction shape {parameters.shape}')
        valid = np.isfinite(parameters).all(axis=1)
        return parameters[valid], valid, diagnostics

    scaling = contract['knot_scaling']
    try:
        knots, controls, _ = prepare_local(parameters, scaling, diagnostics=diagnostics)
        start, end = knots[3], knots[-4]
        valid = ((times - observation) >= start) & ((times - observation) <= end)
        diagnostics['requested_interval_relative'] = (times[[0, -1]] - observation).tolist()
        diagnostics['coverage_valid'] = bool(start < end and valid.all())
        # An endpoint-only probe validates multiplicity/domain even without overlap.
        commands, _ = evaluate_local(knots, controls, times[valid] if valid.any() else [start],
                                      observation if valid.any() else 0,
                                      contract['normalization'], boundary_scale=None)
        diagnostics['decoder_success'] = True
        return commands if valid.any() else commands[:0], valid, diagnostics
    except ValueError as exc:
        diagnostics.update(decoder_success=False, exception=str(exc))
        return np.empty((0, 14)), np.zeros(len(times), dtype=bool), diagnostics


def error_totals(fk, prediction, target):
    if not len(prediction):
        return dict(samples=0, position_squared=0., orientation_sum=0., joint_squared=0., gripper_squared=0.)
    predicted, expected = fk.poses(prediction), fk.poses(target)
    angle = (rotations(expected).inv() * rotations(predicted)).magnitude()
    difference = prediction - target
    return dict(samples=len(prediction), position_squared=float(np.square(predicted[..., :3] - expected[..., :3]).sum()),
                orientation_sum=float(angle.sum()), joint_squared=float(np.square(difference[:, ARM_CHANNELS]).sum()),
                gripper_squared=float(np.square(difference[:, [6, 13]]).sum()))


def aggregate_errors(windows):
    fields = ('ee_position_rmse_m', 'ee_orientation_mean_rad', 'joint_rmse_rad', 'gripper_rmse_fraction')
    count = sum(w['samples'] for w in windows)
    if not count:
        return dict(samples=count, **dict.fromkeys(fields))
    return dict(samples=count, **dict(zip(fields, (
        np.sqrt(sum(w['position_squared'] for w in windows) / (2 * count)),
        sum(w['orientation_sum'] for w in windows) / (2 * count),
        np.sqrt(sum(w['joint_squared'] for w in windows) / (12 * count)),
        np.sqrt(sum(w['gripper_squared'] for w in windows) / (2 * count)),
    ))))


def summarize(windows):
    full = [w for w in windows if w['supported']['samples'] == len(w['source_rows'])]
    supported = sum(w['supported']['samples'] for w in windows)
    requested = sum(len(w['source_rows']) for w in windows)
    return dict(windows=len(windows), full_windows=len(full), full_window_coverage=len(full) / len(windows),
                requested_samples=requested, supported_samples=supported, timestamp_coverage=supported / requested,
                full_window_errors=aggregate_errors([w['supported'] for w in full]),
                supported_errors=aggregate_errors([w['supported'] for w in windows]))


def across_demonstrations(demos):
    result = {}
    for field in ('full_window_coverage', 'timestamp_coverage'):
        values = [demo[field] for demo in demos]
        result[field] = dict(mean=float(np.mean(values)), std=float(np.std(values)), demonstrations=len(values))
    for group in ('full_window_errors', 'supported_errors'):
        result[group] = {}
        for field in demos[0][group]:
            if field == 'samples':
                continue
            values = [d[group][field] for d in demos if d[group][field] is not None]
            result[group][field] = dict(mean=float(np.mean(values)) if values else None,
                std=float(np.std(values)) if values else None, demonstrations=len(values))
    return result


def run_offline(checkpoint, metadata, output, limit, fk):
    import torch
    from omegaconf import OmegaConf
    from data.dataset import collate_fn, create_dataset
    from train.sample import inference_sample

    contract = metadata['contract']
    config = OmegaConf.create(metadata['config'])
    manifest = Path(config.research.window_manifest)
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if manifest_hash != contract['window_manifest_sha256']:
        raise ValueError('Validation window manifest differs from the checkpoint contract')
    dataset = create_dataset(config, val=True)
    windows = dataset.research_windows
    actual = {Path(w['qpos_path']).resolve() for w in windows}
    expected = {Path(p).resolve() for p in contract['dataset_split']['validation_paths']}
    if actual != expected or len(actual) != contract['dataset_split']['validation_episodes']:
        raise ValueError('Validation windows do not cover the checkpoint’s held-out demonstrations')
    model, config, _, _ = load_checkpoint(checkpoint)
    records = defaultdict(list)
    with (output / 'windows.jsonl').open('x') as stream:
        for index, window in enumerate(windows[:limit]):
            rows = window['_dense_indices']
            obs = window['observation_index']
            episode = window['qpos_path']
            seed = index
            record = dict(episode=episode, observation_index=obs, source_rows=rows.tolist(),
                          window_identity=window['identity'], noise_seed=seed)
            torch.manual_seed(seed)
            try:
                with torch.inference_mode():
                    _, prediction = inference_sample(model, collate_fn([dataset[index]]), config)
                commands, valid, diagnostics = score_prediction(
                    prediction[0].float().cpu().numpy(), rows, obs, contract)
                if diagnostics:
                    record['spline_diagnostics'] = diagnostics
                if not valid.all():
                    record['incomplete_reason'] = diagnostics.get(
                        'exception', 'unsupported timestamps' if diagnostics else 'nonfinite dense actions')
            except (ValueError, RuntimeError) as exc:
                commands, valid = np.empty((0, 14)), np.zeros(len(rows), dtype=bool)
                record['incomplete_reason'] = f'{type(exc).__name__}: {exc}'
            record['supported_mask'] = valid.tolist()
            record['supported'] = error_totals(fk, commands, window['_actions'][rows[valid]])
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            records[episode].append(record)
    demos = [dict(episode=episode, **summarize(rows)) for episode, rows in records.items()]
    write_json(output / 'demonstrations.json', demos)
    return dict(**summarize([row for rows in records.values() for row in rows]),
                window_manifest_sha256=manifest_hash, demonstrations=len(demos),
                available_demonstrations=len(expected), available_windows=len(windows),
                across_demonstrations=across_demonstrations(demos))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=['A', 'C'])
    parser.add_argument('--limit-windows', type=int)
    parser.add_argument('--fk-check', action='store_true', help='Run only the RoboTwin FK parity check')
    args = parser.parse_args()
    if (not args.fk_check and not args.checkpoint) or (args.limit_windows is not None and args.limit_windows < 1):
        parser.error('Offline model scoring needs --checkpoint and a positive window limit if specified')
    if not args.fk_check:
        import torch
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME') or not torch.cuda.is_available():
            raise RuntimeError('Offline model scoring requires an allocated Slurm GPU')
    output = args.output.resolve()
    checkpoint = args.checkpoint.resolve(strict=True) if args.checkpoint else None
    metadata = json.loads((checkpoint / 'research_config.json').read_text()) if checkpoint else None
    if metadata and args.variant and metadata['contract']['representation'] != {'A': 'dense', 'C': 'spline'}[args.variant]:
        parser.error('Checkpoint representation does not match the selected variant')
    output.mkdir(parents=True, exist_ok=False)
    cwd = Path.cwd()
    os.chdir(ROBOTWIN_ROOT)
    try:
        api = robotwin_api()
        task, _ = api.load_task_args(dict(task_name='place_can_basket', task_config='demo_clean', policy_name='Motus_AC'))
        fk = RoboTwinFK(task)
        report = dict(status='running', fk_check=fk.check())
    finally:
        os.chdir(cwd)
    if not args.fk_check:
        report.update(checkpoint=str(checkpoint), representation=metadata['contract']['representation'])
        write_json(output / 'results.json', report)
        try:
            report.update(run_offline(checkpoint, metadata, output, args.limit_windows, fk))
        except Exception as exc:
            report.update(status='error', error=f'{type(exc).__name__}: {exc}')
            write_json(output / 'results.json', report)
            raise
    report['status'] = 'complete'
    write_json(output / 'results.json', report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
