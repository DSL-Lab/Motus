"""CPU numerical and executor-contract checks; these are not simulator rollouts."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from inference.robotwin.spline_actions import (
    GRIPPERS,
    JOINTS,
    ActionAdapter,
    boundary_jumps,
    fit_cubic,
    source_schedule,
)
from inference.robotwin.spline_rollout import RolloutRecorder


def trajectory():
    t = np.arange(16, dtype=float) / 15
    actions = np.zeros((16, 14), dtype=np.float32)
    actions[:, JOINTS] = np.array(
        [(i + 1) * (t**3 - 0.4 * t**2 + 0.2 * t) for i in range(12)]
    ).T
    actions[:, 6] = np.where(t < 0.4, 0.12, 0.87)
    actions[:, 13] = np.where(t < 0.8, 0.76, 0.23)
    return t, actions


class SplineChecks(unittest.TestCase):
    def test_dense_one_is_bit_exact_including_signed_zero(self):
        times, actions = trajectory()
        actions[0, 0] = -0.0
        output, info = ActionAdapter().adapt(actions, times)
        self.assertEqual(output.tobytes(), actions.tobytes())
        self.assertEqual(info["processing_s"] >= 0, True)

    def test_cubic_reconstruction_shared_knots_and_gripper_preservation(self):
        times, actions = trajectory()
        output, info = ActionAdapter("spline", joint_tolerances=1e-5).adapt(
            actions, times
        )
        np.testing.assert_allclose(output[:, JOINTS], actions[:, JOINTS], atol=1e-5)
        np.testing.assert_array_equal(output[:, GRIPPERS], actions[:, GRIPPERS])
        self.assertEqual(info["control_points"], 4)
        self.assertFalse(info["fallback"])
        self.assertGreater(info["storage_ratio_dense_over_representation"], 1)

    def test_adaptive_shared_knots_meet_each_channel_tolerance(self):
        times, actions = trajectory()
        actions[:, 0] += 0.3 * np.exp(-(((times - 0.4) / 0.1) ** 2))
        tolerances = np.full(12, 1e-4)
        fit = fit_cubic(times, actions[:, JOINTS], tolerances)
        self.assertIsNotNone(fit.spline, fit.reason)
        self.assertGreater(len(fit.spline.c), 4)
        self.assertTrue(np.all(fit.max_error <= tolerances))

    def test_fit_failure_falls_back_exactly_and_records_failure(self):
        times, actions = trajectory()
        actions[:, 0] = np.random.default_rng(7).normal(size=16)
        for speed in (1.0, 1.5, 2.0):
            actual, info = ActionAdapter("spline", speed, 1e-8, 4).adapt(actions, times)
            dense, _ = ActionAdapter("dense", speed).adapt(actions, times)
            self.assertTrue(info["fallback"])
            self.assertIsNotNone(info["fit_failure"])
            self.assertEqual(actual.tobytes(), dense.tobytes())
            json.dumps(info, allow_nan=False)

    def test_per_channel_validation_cannot_hide_one_bad_joint(self):
        times, actions = trajectory()
        actions[:, 7] = np.sin(times * 13)
        tolerances = np.full(12, 100.0)
        tolerances[6] = 1e-9
        fit = fit_cubic(times, actions[:, JOINTS], tolerances, 4)
        self.assertIsNone(fit.spline)
        self.assertGreater(fit.max_error[6], tolerances[6])

    def test_same_map_for_dense_spline_and_grippers_at_all_speeds(self):
        times, actions = trajectory()
        # Linear continuous channels make dense and cubic evaluation identical.
        actions[:, JOINTS] = np.arange(12)[None] * times[:, None]
        for speed in (1.0, 1.5, 2.0):
            dense, dense_info = ActionAdapter("dense", speed).adapt(actions, times, 12)
            spline, spline_info = ActionAdapter("spline", speed).adapt(
                actions, times, 12
            )
            np.testing.assert_allclose(spline[:, JOINTS], dense[:, JOINTS], atol=2e-6)
            np.testing.assert_array_equal(spline[:, GRIPPERS], dense[:, GRIPPERS])
            query = np.array(dense_info["source_query"])
            self.assertEqual(dense_info["source_query"], spline_info["source_query"])
            self.assertEqual(query[0], times[0])
            self.assertEqual(query[-1], times[11])
            self.assertTrue(np.all(query <= times[11]))
            indices = np.searchsorted(times, query, side="right") - 1
            np.testing.assert_array_equal(
                dense[:, GRIPPERS], actions[indices][:, GRIPPERS]
            )

    def test_endpoint_rounding_never_wraps_or_extends_prefix(self):
        for n in range(2, 17):
            for speed in (1.0, 1.5, 2.0):
                t = np.arange(1, n + 1) * 0.18
                query = source_schedule(t, speed)
                self.assertAlmostEqual(query[-1], t[-1])
                self.assertTrue(np.all(np.diff(query) > 0))
                self.assertEqual(len(query), int(np.ceil((n - 1) / speed)) + 1)
                np.testing.assert_allclose(
                    query[1:-1] - t[0], np.arange(1, len(query) - 1) * 0.18 * speed
                )

    def test_analog_gripper_changes_collapsed_in_one_slot_are_reported(self):
        times, actions = trajectory()
        actions[:, GRIPPERS] = np.arange(16)[:, None] / 20
        _, info = ActionAdapter("dense", 2.0).adapt(actions, times)
        self.assertGreater(info["collapsed_gripper_changes"], 0)

    def test_retiming_hits_exact_original_event_times_with_large_clock_offset(self):
        _, actions = trajectory()
        times = 98765.4 + np.arange(16) * 0.06
        actions[:, GRIPPERS] = np.arange(16)[:, None] / 20
        for speed in (1.5, 2.0):
            output, info = ActionAdapter("dense", speed).adapt(actions, times)
            positions = np.minimum(np.arange(len(output)) * speed, 15)
            np.testing.assert_array_equal(
                output[:, GRIPPERS], actions[positions.astype(int)][:, GRIPPERS]
            )
            for index, position in enumerate(positions):
                if position == int(position):
                    self.assertEqual(info["source_query"][index], times[int(position)])

    def test_invalid_semantics_timestamps_and_nonfinite_are_rejected(self):
        times, actions = trajectory()
        with self.assertRaises(ValueError):
            ActionAdapter(action_semantics="delta_pose")
        with self.assertRaises(ValueError):
            ActionAdapter().adapt(actions, times**2)
        with self.assertRaises(ValueError):
            ActionAdapter(joint_tolerances=0)
        with self.assertRaises(ValueError):
            source_schedule(times, 0)
        actions[0, 0] = np.nan
        with self.assertRaises(ValueError):
            ActionAdapter("spline").adapt(actions, times)

    def test_boundary_diagnostics_do_not_align_actions(self):
        _, actions = trajectory()
        previous = actions[-2:].copy()
        current = actions[:2].copy()
        copy = current.copy()
        info = boundary_jumps(previous, current)
        self.assertNotIn("velocity_jump_per_joint", info)
        np.testing.assert_array_equal(current, copy)
        info = boundary_jumps(previous, current, 0.1)
        self.assertEqual(len(info["velocity_jump_per_joint"]), 12)


class FakeScene:
    def __init__(self):
        self.steps = 0

    def get_timestep(self):
        return 1 / 250

    def step(self):
        self.steps += 1


class FakeEnvironment:
    def __init__(self, stop):
        self.scene = FakeScene()
        self.ep_num = 0
        self.eval_success = False
        self.take_action_cnt = 0
        self.step_lim = 20
        self.stop = stop
        self.accepted = []

    def get_obs(self):
        return {}

    def take_action(self, action, action_type):
        if self.eval_success or self.take_action_cnt == self.step_lim:
            return
        self.take_action_cnt += 1
        self.accepted.append(action.copy())
        # Deliberately variable target durations, like TOPP.
        for _ in range(self.take_action_cnt):
            self.scene.step()
        self.eval_success = self.take_action_cnt == self.stop


class RecorderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary_root = Path(__file__).resolve().parents[1] / ".cache"
        cls.temporary_root.mkdir(exist_ok=True)

    def test_original_actions_physics_counts_and_early_success_preserved(self):
        _, actions = trajectory()

        def evaluate(env, model, observation):
            for action in model.get_action():
                env.take_action(action, action_type="qpos")

        for mode in ("original", "dense", "spline"):
            with TemporaryDirectory(dir=self.temporary_root) as temporary:
                model = SimpleNamespace(get_action=lambda: actions)
                policy = SimpleNamespace(
                    get_model=lambda args, model=model: model,
                    reset_model=lambda model: None,
                    eval=evaluate,
                )
                env = FakeEnvironment(stop=3)
                scene = env.scene
                recorder = RolloutRecorder(policy, temporary, mode, 1.0, 0.01, 16)
                recorder.attach_environment(env)
                recorder.get_model({})
                recorder.reset_model(model)
                obs = env.get_obs()
                recorder.eval(env, model, obs)
                self.assertIs(env.scene, scene)
                self.assertEqual(scene.steps, 6)
                self.assertEqual(len(env.accepted), 3)
                self.assertEqual(recorder.episodes[0]["physics_steps"], 6)
                self.assertAlmostEqual(recorder.episodes[0]["simulated_s"], 6 / 250)
                self.assertEqual(recorder.episodes[0]["policy_calls"], 1)
                chunk = json.loads((Path(temporary) / "chunks.jsonl").read_text())
                self.assertGreaterEqual(chunk["observation_to_action_s"], 0)
                if mode != "spline":
                    self.assertEqual(
                        np.array(env.accepted).tobytes(), actions[:3].tobytes()
                    )

    def test_step_limit_remains_failure(self):
        _, actions = trajectory()
        env = FakeEnvironment(stop=999)
        env.step_lim = 2
        with TemporaryDirectory(dir=self.temporary_root) as temporary:
            model = SimpleNamespace(get_action=lambda: actions)

            def evaluate(task, model, observation):
                for action in model.get_action():
                    task.take_action(action, action_type="qpos")

            policy = SimpleNamespace(
                get_model=lambda args: model,
                reset_model=lambda model: None,
                eval=evaluate,
            )
            recorder = RolloutRecorder(policy, temporary, "dense", 1.0, 0.01, 16)
            recorder.attach_environment(env)
            recorder.get_model({})
            recorder.reset_model(model)
            recorder.eval(env, model, env.get_obs())
            self.assertFalse(recorder.episodes[0]["success"])
            self.assertEqual(env.take_action_cnt, 2)


if __name__ == "__main__":
    unittest.main()
