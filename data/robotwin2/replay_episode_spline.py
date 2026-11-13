"""Paired expert-path replay using RoboTwin's saved seed/path and shared decoder.

Run on the simulator host, from WAM-RoboTwin's root. No policy/model loaded.
The baseline traces the existing play_once/load_tran_data path without saving
images. The second run substitutes decoded position/gripper commands at exactly
the same scene steps, retaining original velocity feedforward and gripper eps.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from adaptive_spline import (decode_episode_grid, identity, read_cache)


class StepClock:
    def __init__(self, scene):
        self.scene, self.steps = scene, 0
        self.after_step = lambda: None

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def step(self):
        self.scene.step()
        self.steps += 1
        self.after_step()


def snapshot(env):
    return dict(robot=env.robot.get_left_arm_real_jointState() + env.robot.get_right_arm_real_jointState(),
                actors=[dict(name=a.get_name(), p=a.get_pose().p.tolist(), q=a.get_pose().q.tolist())
                        for a in env.scene.get_all_actors()])


def run_trial(task_factory, args, episode, seed, commands=None, expected_initial=None):
    """Intercept existing replay setters only; do not replace simulator stepping."""
    random.seed(seed)
    env = task_factory(args["task_name"])
    env.setup_demo(now_ep_num=episode, seed=seed, **args)
    try:
        env.set_path_lst(dict(args, **env.load_tran_data(episode)))
        initial = snapshot(env)
        if expected_initial is not None and identity(initial) != identity(expected_initial):
            raise ValueError("Initial robot/actor snapshots differ before replay execution")
        clock = StepClock(env.scene)
        env.scene = clock
        pictures, picture_steps, measured, poses, events, commanded = [], [], [], [], [], []
        def picture():
            pictures.append(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState())
            picture_steps.append(clock.steps)
        env._take_picture = picture  # replaces only recording; preserves call sites
        original_arm, original_gripper = env.robot.set_arm_joints, env.robot.set_gripper
        def arm(position, velocity, arm_tag):
            events.append([clock.steps, "arm", str(arm_tag)])
            if commands is not None:
                if clock.steps >= len(commands):
                    raise ValueError("Replay requests commands beyond verified timing coverage")
                position = commands[clock.steps, :6] if arm_tag == "left" else commands[clock.steps, 7:13]
            return original_arm(position, velocity, arm_tag)
        def gripper(value, arm_tag, gripper_eps=0.1):
            events.append([clock.steps, "gripper", str(arm_tag)])
            if commands is not None:
                if clock.steps >= len(commands):
                    raise ValueError("Replay requests gripper beyond verified timing coverage")
                value = commands[clock.steps, 6 if arm_tag == "left" else 13]
            return original_gripper(value, arm_tag, gripper_eps)
        env.robot.set_arm_joints, env.robot.set_gripper = arm, gripper
        def capture():
            measured.append(env.robot.get_left_arm_real_jointState() + env.robot.get_right_arm_real_jointState())
            commanded.append(env.robot.get_left_arm_jointState() + env.robot.get_right_arm_jointState())
            poses.append([env.get_arm_pose("left")[:3], env.get_arm_pose("right")[:3]])
        clock.after_step = capture
        env.play_once()
        return dict(initial=initial, picture_steps=np.asarray(picture_steps), pictures=np.asarray(pictures),
                    measured=np.asarray(measured), commanded=np.asarray(commanded), poses=np.asarray(poses), events=events,
                    steps=clock.steps, timestep=float(clock.get_timestep()), success=bool(env.check_success()))
    finally:
        env.close_env()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--robotwin-root", type=Path, required=True)
    p.add_argument("--recording-dir", type=Path, required=True, help="Original directory with seed.txt and _traj_data/")
    p.add_argument("--task-config", default="demo_clean", help="Must be the original collection config")
    p.add_argument("--episode-index", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--knot-scaling", type=Path)
    p.add_argument("--diagnostic-only", action="store_true")
    args = p.parse_args(argv)
    output = args.output.resolve()
    if output.suffix != ".json":
        p.error("--output must name a .json report")
    if output.exists() or output.with_suffix(".npz").exists():
        p.error("Replay output exists; choose a fresh path")
    actions, spline, meta = read_cache(args.cache)
    if not meta["diagnostics"]["success"]:
        p.error("Stage-1 fit failed")
    if meta["normalization"]["scope"] == "diagnostic-only" and not args.diagnostic_only:
        p.error("Diagnostic normalization requires --diagnostic-only")
    if args.knot_scaling:
        scaling = json.loads(args.knot_scaling.read_text())
    elif args.diagnostic_only:
        scaling = dict(scope="diagnostic-only", units="source_action_step", offset=0., scale=100.)
    else:
        p.error("Supply trained --knot-scaling or explicit --diagnostic-only")
    if scaling["scope"] == "diagnostic-only" and not args.diagnostic_only:
        p.error("Diagnostic scaling requires --diagnostic-only")
    if scaling["scope"] == "training-only" and scaling.get("normalization_identity") != meta["normalization"]["identity"]:
        p.error("Knot/action normalization identities disagree")
    root, recording = args.robotwin_root.resolve(strict=True), args.recording_dir.resolve(strict=True)
    if Path.cwd().resolve() != root:
        p.error("Run from --robotwin-root to preserve existing RoboTwin asset paths")
    if not (recording / "_traj_data" / f"episode{args.episode_index}.pkl").exists():
        p.error("Saved expert path missing; cannot reproduce initial scene/timing")
    seeds = [int(v) for v in (recording / "seed.txt").read_text().split()]
    if not 0 <= args.episode_index < len(seeds):
        p.error("Episode index outside saved seed list")
    sys.path.insert(0, str(root))
    # Heavy simulator imports occur only after preflight. Reuse collection helpers.
    import yaml
    from scripts.collect_data import class_decorator, get_embodiment_config
    config_dir = root / "env_cfg" / "task_config"
    config = yaml.safe_load((config_dir / (args.task_config + ".yml")).read_text())
    embodiments = yaml.safe_load((config_dir / "_embodiment_config.yml").read_text())
    if config["embodiment"] != ["aloha-agilex"]:
        p.error("This audited replay entry supports the original aloha-agilex embodiment only")
    robot_file = embodiments["aloha-agilex"]["file_path"]
    config.update(task_name=meta["task"], task_config=args.task_config, save_path=str(recording),
        save_data=False, need_plan=False, eval_mode=False, render_freq=0,
        left_robot_file=robot_file, right_robot_file=robot_file, dual_arm_embodied=True,
        left_embodiment_config=get_embodiment_config(robot_file),
        right_embodiment_config=get_embodiment_config(robot_file))
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status="started", replay_passed=False, source=meta["source"],
                  seed=seeds[args.episode_index], episode=args.episode_index, scaling=scaling,
                  controller="existing saved-path play_once scene-step schedule; original velocity feedforward and gripper eps",
                  simulator_config=config, command_units="joint radians; continuous gripper fractions")
    try:
        baseline = run_trial(class_decorator, config, args.episode_index, report["seed"])
        if baseline["pictures"].shape != actions.shape or not np.allclose(baseline["pictures"], actions, atol=2e-6, rtol=0):
            raise ValueError("Expert replay does not reproduce source commanded rows; wrong seed/config/assets or episode")
        ticks = baseline["picture_steps"]
        if ticks[0] != 0 or ticks[-1] != baseline["steps"] or np.any(np.diff(ticks) < 0):
            raise ValueError("Recorded picture clock does not cover the entire controller trajectory")
        for tick in np.unique(ticks):
            same = actions[ticks == tick]
            if not np.allclose(same, same[0], atol=2e-6, rtol=0):
                raise ValueError("Different source commands at an identical simulator timestamp; ambiguous spline time mapping")
        # Duplicate boundary images have identical commands. Explicitly choose
        # the last recorded row at that timestamp when inverting the clock.
        unique_ticks, reverse = np.unique(ticks[::-1], return_index=True)
        rows = len(ticks) - 1 - reverse
        grid = np.interp(np.arange(baseline["steps"] + 1), unique_ticks, rows)
        decoded, decoder_reports = decode_episode_grid(spline, grid, scaling, meta["normalization"])
        reconstructed = run_trial(class_decorator, config, args.episode_index, report["seed"], decoded, baseline["initial"])
        if identity(baseline["initial"]) != identity(reconstructed["initial"]):
            raise ValueError("Initial robot/actor snapshots differ across paired replays")
        if baseline["events"] != reconstructed["events"] or baseline["steps"] != reconstructed["steps"]:
            raise ValueError("Paired replay controller event schedule changed")
        dt = baseline["timestep"]
        if not np.isfinite(dt) or dt <= 0 or dt != reconstructed["timestep"]:
            raise ValueError("Invalid or changed live simulator timestep")
        timeline = dict(status="verified", units="seconds", action_sha256=meta["source"]["action_sha256"],
                        action_times=(ticks * dt).tolist(), camera_times=(ticks * dt).tolist(),
                        controller_times=(np.arange(baseline["steps"] + 1) * dt).tolist(),
                        duplicate_timestamp_rule="last identical-command row", provenance="matched saved-seed/path expert replay")
        output.with_suffix(".timeline.json").write_text(json.dumps(timeline, indent=2))
        report.update(status="completed", replay_passed=baseline["success"] and reconstructed["success"], initial_states_matched=True,
            original_success=baseline["success"], reconstructed_success=reconstructed["success"],
            scene_steps=baseline["steps"], timestep_seconds=dt,
            joint_tracking_difference_rmse=float(np.sqrt(np.mean((baseline["measured"][:, [*range(6), *range(7,13)]] - reconstructed["measured"][:, [*range(6), *range(7,13)]])**2))),
            end_effector_position_difference_rmse_m=float(np.sqrt(np.mean(np.sum((baseline["poses"] - reconstructed["poses"])**2, axis=-1)))),
            gripper_out_of_range_count=int(np.count_nonzero((decoded[:, [6,13]] < 0) | (decoded[:, [6,13]] > 1))),
            total_decoder_boundary_snaps=sum(r["boundary_roundoff_snaps"] for r in decoder_reports))
        report["per_channel_executed_command_difference_rmse"] = np.sqrt(np.mean((baseline["commanded"] - reconstructed["commanded"])**2, axis=0)).tolist()
        report["gripper_midpoint_crossing_steps_diagnostic_only"] = {
            name: [np.flatnonzero(np.diff(values[:, j] >= .5)).tolist() for j in [6, 13]]
            for name, values in [("original", baseline["commanded"]), ("reconstructed", reconstructed["commanded"])]}
        np.savez_compressed(output.with_suffix(".npz"), original_measured=baseline["measured"],
            reconstructed_measured=reconstructed["measured"], original_poses=baseline["poses"],
            reconstructed_poses=reconstructed["poses"], decoded_commands=decoded, controller_source_times=grid,
            original_executed_commands=baseline["commanded"], reconstructed_executed_commands=reconstructed["commanded"])
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != "simulator_config"}, indent=2))
    return 0 if report["replay_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
