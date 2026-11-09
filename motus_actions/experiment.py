"""Small RoboTwin experiment hooks; logging stays outside the spline mathematics."""

import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np

from .robotwin import ARM_CHANNELS, GRIPPER_CHANNELS, adapt_chunk
from .spline import save_spline
from .tcp_metrics import RobotTCPKinematics, chunk_position_metrics, pool_statistics


def configure(model, arguments):
    tolerances = arguments.get("fitting_tolerances", [0.01] * 12)
    model.action_options = {
        "action_adapter": arguments.get("action_adapter", "dense"),
        "speed_factor": float(arguments.get("speed_factor", 1.0)),
        "source_dt": float(arguments.get("source_dt", 0.1)),
        "tolerances": tolerances,
        "smoothing": float(arguments.get("fitting_smoothing", 1.0)),
        "max_control_points": arguments.get("max_control_points"),
    }
    # Validate configuration even if the first episode never reaches inference.
    adapt_chunk(np.zeros((4, 14)), np.zeros(14), **model.action_options)
    model.experiment_config = dict(arguments)
    model.experiment_dir = Path(arguments.get("log_dir") or "logs/motus_actions")
    model.experiment_dir.mkdir(parents=True, exist_ok=True)
    model.policy_seed = arguments.get("policy_seed")
    model.episode_metrics = None
    model.previous_command = None
    dense_predict = model.get_action

    def timed_predict(*args, **kwargs):
        _synchronize(model)
        start = perf_counter()
        metrics = model.episode_metrics
        if metrics is not None:
            metrics["policy_calls"] += 1
        try:
            actions = dense_predict(*args, **kwargs)
            _synchronize(model)
        finally:
            if metrics is not None:
                latency = perf_counter() - start
                metrics["inference_seconds"] += latency
                metrics["inference_latencies"].append(latency)
        if metrics is not None:
            metrics["dense_sample_counts"].append(len(actions))
        return actions

    # Measures --original too, while leaving its upstream execution function intact.
    model.get_action = timed_predict


def begin_episode(model, env, metadata):
    if model.policy_seed is not None:
        import torch

        random.seed(int(model.policy_seed))
        np.random.seed(int(model.policy_seed))
        torch.manual_seed(int(model.policy_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(model.policy_seed))
    model.previous_command = None
    model.episode_metrics = dict(
        **metadata,
        configuration=model.experiment_config,
        action_options=model.action_options,
        policy_seed=model.policy_seed,
        chunks=[],
        policy_calls=0,
        inference_seconds=0.0,
        inference_latencies=[],
        dense_sample_counts=[],
        fit_seconds=0.0,
        decode_seconds=0.0,
        fallback_count=0,
        failure_notes=[],
    )
    model.episode_wall_start = perf_counter()
    model.episode_sim_start = env.live_sim_step
    model.tcp_kinematics = None
    if getattr(model, "_original_eval", None) is None:
        try:
            model.tcp_kinematics = RobotTCPKinematics(env.robot)
            model.episode_metrics["tcp_kinematics"] = model.tcp_kinematics.metadata
            model.episode_metrics["tcp_fk_check_m"] = (
                model.tcp_kinematics.verify_live_tcp(env.robot)
            )
        except Exception as error:  # noqa: BLE001 -- diagnostics must not change execution
            model.tcp_kinematics = None
            model.episode_metrics["tcp_metrics_unavailable"] = (
                f"{type(error).__name__}: {error}"
            )


def _synchronize(model):
    # Keep CUDA work inside measured latency without modifying inference or RNG.
    if str(model.device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(model.device)


def eval_chunk(env, model, observation):
    metrics = model.episode_metrics
    if metrics is None:
        raise RuntimeError("begin_episode must run before evaluating a chunk")
    model.set_instruction(env.get_instruction())
    model.update_obs(observation)
    actions = (
        model.get_action()
    )  # original joint future-video/action inference, unchanged
    latency = metrics["inference_latencies"][-1]
    state = model.current_state.detach().float().cpu().numpy().reshape(14)
    commands, spline, info = adapt_chunk(actions, state, **model.action_options)
    info["inference_seconds"] = latency
    info["boundary_jump_max_rad"] = (
        None
        if model.previous_command is None
        else float(
            np.max(
                np.abs(commands[0, ARM_CHANNELS] - model.previous_command[ARM_CHANNELS])
            )
        )
    )
    info["initial_tracking_jump_max_rad"] = float(
        np.max(np.abs(commands[0, ARM_CHANNELS] - state[ARM_CHANNELS]))
    )
    info["command_physics_steps"] = []
    info["executed_commands"] = 0
    metrics["chunks"].append(info)
    for key in ("fit_seconds", "decode_seconds", "fallback_count"):
        metrics[key] += info[key]
    if info["failure"]:
        metrics["failure_notes"].append(info["failure"])
    chunk_id = len(metrics["chunks"]) - 1
    prefix = model.experiment_dir / f"seed{metrics['scene_seed']}_chunk{chunk_id:04d}"
    # Preserve dense output, source/query timestamps, and discrete gripper samples.
    np.savez(
        str(prefix) + "_actions.npz",
        dense_actions=actions,
        initial_state=state,
        source_times=info["source_times"],
        query_source_times=info["query_source_times"],
        execution_commands=commands,
        gripper_source_times=info["source_times"],
        gripper_values=np.vstack([state, actions])[:, GRIPPER_CHANNELS],
        gripper_event_source_times=info["gripper_event_source_times"],
    )
    if spline is not None:
        save_spline(str(prefix) + "_spline.npz", spline, info["source_times"])
    for action in commands:
        before = env.live_sim_step
        env.take_action(action, action_type="qpos")
        advanced = env.live_sim_step - before
        info["command_physics_steps"].append(int(advanced))
        # Preserve original no-op take_action calls after early success/limit.
        if advanced:
            info["executed_commands"] += 1
            model.previous_command = np.array(action, copy=True)
    if model.tcp_kinematics is not None:
        start = perf_counter()
        try:
            info["tcp_position_metrics"] = chunk_position_metrics(
                model.tcp_kinematics,
                actions,
                state,
                info["source_times"],
                info["query_source_times"],
                commands,
                spline,
                info,
                info["command_physics_steps"],
            )
        except Exception as error:  # noqa: BLE001 -- diagnostics must not change fallback/success
            info["tcp_metrics_unavailable"] = f"{type(error).__name__}: {error}"
        info["tcp_metric_seconds"] = perf_counter() - start


def end_episode(model, env, metadata):
    metrics = model.episode_metrics
    if metrics is None:
        return None
    metrics.update(metadata)
    metrics["wall_seconds"] = perf_counter() - model.episode_wall_start
    metrics["physics_steps"] = int(env.live_sim_step - model.episode_sim_start)
    metrics["physics_dt"] = float(env.scene.get_timestep())
    metrics["simulated_seconds"] = metrics["physics_steps"] * metrics["physics_dt"]
    metrics["action_calls"] = int(env.take_action_cnt)
    tcp_chunks = [
        chunk["tcp_position_metrics"]
        for chunk in metrics["chunks"]
        if "tcp_position_metrics" in chunk
    ]
    metrics["tcp_position_metrics"] = {
        "timestamp_convention": "Nominal source times; source includes t=0 measured fit anchor; query reference is dense joint np.interp then FK. Query times are not simulator timestamps.",
        "sampling_limit": "Saved source/query samples only; no continuous-time or controller tracking bound.",
        "tracking": {
            "available": False,
            "reason": "Measured TCP/controller references at actual simulator timestamps are not saved",
        },
        "dispatched_query": pool_statistics(
            [chunk["dispatched_query"] for chunk in tcp_chunks]
        ),
        "dispatched_query_physics_positive": pool_statistics(
            [chunk["dispatched_query_physics_positive"] for chunk in tcp_chunks]
        ),
        "candidates": {
            status: {
                key: pool_statistics(
                    [
                        chunk[key]
                        for chunk in tcp_chunks
                        if chunk["candidate_status"] == status
                    ]
                )
                for key in (
                    "candidate_source",
                    "candidate_query",
                    "candidate_query_physics_positive",
                )
            }
            for status in ("accepted", "rejected")
        },
        "measured_chunks": len(tcp_chunks),
        "total_chunks": len(metrics["chunks"]),
    }
    if metadata.get("failure"):
        metrics["failure_notes"].append(metadata["failure"])
    with (model.experiment_dir / "episodes.jsonl").open("a") as stream:
        stream.write(json.dumps(metrics, allow_nan=False) + "\n")
    model.episode_metrics = None
    model.previous_command = None
    model.tcp_kinematics = None
    return metrics
