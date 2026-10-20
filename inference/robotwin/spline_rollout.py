"""Instrument the existing Motus wrapper without editing either policy or simulator."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from inference.robotwin.spline_actions import ActionAdapter, boundary_jumps


def append_record(path, record):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


class SceneClock:
    """Forward the real scene API and count its actual physics steps."""

    def __init__(self, scene):
        self.scene = scene
        self.steps = 0
        self.seconds = 0.0

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def step(self, *args, **kwargs):
        period = float(self.scene.get_timestep())
        result = self.scene.step(*args, **kwargs)
        self.steps += 1
        self.seconds += period
        return result


class RolloutRecorder:
    def __init__(self, policy, directory, mode, speed_factor, tolerances, max_controls):
        self.policy = policy
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.adapter = (
            None
            if mode == "original"
            else ActionAdapter(mode, speed_factor, tolerances, max_controls)
        )
        self.mode = mode
        self.speed_factor = speed_factor
        self.env = None
        self.episodes = []

    def attach_environment(self, env):
        self.env = env
        original = env.get_obs

        def get_obs(*args, **kwargs):
            started = perf_counter()
            observation = original(*args, **kwargs)
            self.observation_ready = perf_counter()
            self.observation_capture_s = self.observation_ready - started
            return observation

        env.get_obs = get_obs
        return env

    def get_model(self, arguments):
        model = self.policy.get_model(arguments)
        original = model.get_action

        def get_action(*args, **kwargs):
            started = perf_counter()
            actions = original(*args, **kwargs)
            # Original get_action's .cpu().numpy() synchronizes action production.
            policy_s = perf_counter() - started
            source = np.arange(1, len(actions) + 1, dtype=float)
            info = {
                "mode": "original",
                "processing_s": 0.0,
                "fallback": False,
                "dense_bytes": actions.nbytes,
                "representation_bytes": actions.nbytes,
            }
            output = actions
            if self.adapter is not None:
                output, info = self.adapter.adapt(actions, source)
            info.update(boundary_jumps(self.executed[-2:], output))
            artifact = f"episode_{self.env.ep_num:04d}_chunk_{self.calls:04d}.npz"
            np.savez(
                self.directory / artifact,
                predicted=actions,
                submitted=output,
                source_times=source,
            )
            info.update(
                {
                    "episode_index": int(self.env.ep_num),
                    "chunk_index": self.calls,
                    "policy_get_action_s": policy_s,
                    "artifact": artifact,
                    "artifact_bytes": (self.directory / artifact).stat().st_size,
                    "source_time_unit": "action_sample_index",
                    "velocity_jump_per_joint": None,
                }
            )
            self.current_chunk = info
            self.first_action_pending = True
            self.calls += 1
            return output

        model.get_action = get_action
        return model

    def reset_model(self, model):
        self.policy.reset_model(model)
        self.started = perf_counter()
        self.calls = 0
        self.simulated_s = 0.0
        self.physics_steps = 0
        self.executed = np.empty((0, 14))
        self.chunks = []
        self.current_chunk = None
        self.first_action_pending = False
        self.finished = False
        self.observation_ready = None
        self.observation_capture_s = None

    def eval(self, env, model, observation):
        clock = SceneClock(env.scene)
        real_scene = env.scene
        recorder = self

        class TaskProxy:
            def __getattr__(self, name):
                return getattr(env, name)

            def take_action(self, action, *args, **kwargs):
                # Original eval keeps calling take_action after early success;
                # the actual executor returns immediately. Preserve that behavior.
                accepted = not env.eval_success and env.take_action_cnt < env.step_lim
                if accepted and recorder.first_action_pending:
                    recorder.current_chunk["observation_to_action_s"] = (
                        perf_counter() - recorder.observation_ready
                        if recorder.observation_ready is not None
                        else None
                    )
                    recorder.current_chunk["observation_capture_s"] = (
                        recorder.observation_capture_s
                    )
                    recorder.first_action_pending = False
                result = env.take_action(action, *args, **kwargs)
                if accepted:
                    recorder.executed = np.concatenate(
                        (recorder.executed, np.asarray(action)[None])
                    )[-2:]
                return result

        env.scene = clock
        try:
            # Keep original instruction, preprocessing, joint video-action inference,
            # frame saving, action ordering, controller, and early-success behavior.
            self.policy.eval(TaskProxy(), model, observation)
        finally:
            env.scene = real_scene
            self.simulated_s += clock.seconds
            self.physics_steps += clock.steps
        if self.current_chunk is not None:
            self.chunks.append(self.current_chunk)
            append_record(self.directory / "chunks.jsonl", self.current_chunk)
        if not self.finished and (
            env.eval_success or env.take_action_cnt >= env.step_lim
        ):
            record = {
                "episode_index": int(env.ep_num),
                "success": bool(env.eval_success),
                "simulated_s": self.simulated_s,
                "physics_steps": self.physics_steps,
                "wall_s": perf_counter() - self.started,
                "policy_calls": self.calls,
                "action_steps": int(env.take_action_cnt),
                "mode": self.mode,
                "speed_factor": self.speed_factor,
                "spline_processing_s": sum(c["processing_s"] for c in self.chunks),
                "fallback_chunks": sum(c["fallback"] for c in self.chunks),
            }
            self.episodes.append(record)
            append_record(self.directory / "timing.jsonl", record)
            self.finished = True


def install_hooks(namespace, policy, recorder):
    """Patch the evaluation module's extension points, not its task rules."""
    original_factory = namespace["class_decorator"]

    def environment_factory(task_name):
        return recorder.attach_environment(original_factory(task_name))

    def policy_function(policy_name, function_name):
        overrides = {
            "get_model": recorder.get_model,
            "eval": recorder.eval,
            "reset_model": recorder.reset_model,
        }
        return overrides.get(function_name, getattr(policy, function_name, None))

    namespace["class_decorator"] = environment_factory
    namespace["eval_function_decorator"] = policy_function
