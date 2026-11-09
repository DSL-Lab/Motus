import ast
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from motus_actions import experiment, robotwin
from motus_actions.spline import (
    ActionSpline,
    FitError,
    evaluate_spline,
    fit_spline,
    load_spline,
    reconstruction_error,
    save_spline,
)


def joint_chunk(n=16):
    t = np.arange(1, n + 1) * 0.1
    actions = np.zeros((n, 14), dtype=np.float32)
    actions[:, robotwin.ARM_CHANNELS] = t[:, None] ** 2 * np.linspace(0.1, 0.4, 12)
    return actions, np.zeros(14, dtype=np.float32)


def test_timestamped_cubic_multichannel_and_roundtrip(tmp_path):
    # Nonuniform absolute source seconds, twelve coordinated physical channels.
    t = np.array([4.0, 4.02, 4.2, 4.5, 4.7, 5.0, 5.4])
    y = np.column_stack([(t - 4) ** 3 * (i + 1) for i in range(12)])
    spline = fit_spline(t, y, [1e-8] * 12, smoothing=0)
    query = np.array([4.01, 4.37, 5.13])
    expected = np.column_stack([(query - 4) ** 3 * (i + 1) for i in range(12)])
    np.testing.assert_allclose(evaluate_spline(spline, query), expected, atol=1e-12)
    save_spline(tmp_path / "parameters.npz", spline, t)
    restored, timestamps = load_spline(tmp_path / "parameters.npz")
    np.testing.assert_array_equal(timestamps, t)
    np.testing.assert_array_equal(
        evaluate_spline(restored, query), evaluate_spline(spline, query)
    )
    assert restored.source_domain == (4.0, 5.4)


def test_adaptive_shared_knots_and_physical_tolerances():
    t = np.linspace(0, 2, 61)
    y = np.column_stack([np.sin(t * (i + 1) / 2) for i in range(12)])
    tolerance = np.linspace(0.001, 0.01, 12)
    spline = fit_spline(t, y, tolerance)
    error = reconstruction_error(spline, t, y)
    assert np.all(error["max_abs"] <= tolerance)
    assert len(spline.control_points) < len(t)
    assert len(spline.knots) == len(spline.control_points) + 4


def test_maximum_error_is_checked_independently_of_smoothing(monkeypatch):
    # Mock a library result that violates one tight channel while other channels fit.
    t = np.linspace(0, 1, 5)
    coefficients = np.zeros((4, 12))
    coefficients[:, 7] = 1.1  # already in tolerance-scaled fitting coordinates
    bad = SimpleNamespace(t=np.r_[np.zeros(4), np.ones(4)], c=coefficients)
    monkeypatch.setattr("motus_actions.spline.make_splprep", lambda *a, **k: (bad, t))
    with pytest.raises(FitError, match="Maximum reconstruction") as raised:
        fit_spline(t, np.zeros((5, 12)), [0.01] * 12, smoothing=100)
    assert raised.value.errors["max_abs"][7] == pytest.approx(0.011)


def test_control_point_budget_is_enforced():
    t = np.linspace(0, 1, 20)
    with pytest.raises(FitError, match="budget"):
        fit_spline(
            t, np.sin(t[:, None] * 100), [1e-8], smoothing=0, max_control_points=4
        )


@pytest.mark.parametrize("bad_times", [[0, 0, 1, 2], [0, 2, 1, 3], [0, 1, 2, np.nan]])
def test_invalid_timestamps(bad_times):
    with pytest.raises(ValueError):
        fit_spline(bad_times, np.zeros((4, 2)), [0.01, 0.01])


def test_decoder_rejects_invalid_parameters_and_extrapolation():
    with pytest.raises(ValueError, match="counts"):
        ActionSpline(3, np.arange(8), np.zeros((5, 2)), (0, 1))
    spline = fit_spline(np.arange(4), np.zeros((4, 2)), [0.1, 0.1])
    with pytest.raises(ValueError, match="outside"):
        evaluate_spline(spline, [-0.01])
    with pytest.raises(ValueError, match="outside"):
        evaluate_spline(spline, [3.01])


def test_execution_decoder_accepts_direct_parameters_without_fitting():
    slopes = np.linspace(0.1, 1.0, 12)
    direct_parameters = ActionSpline(
        3,
        np.r_[np.zeros(4), np.ones(4)],
        np.linspace(0, 1, 4)[:, None] * slopes,
        (0.0, 1.0),
    )
    query = np.array([0.25, 0.5, 0.75])
    gripper_times = np.array([0.0, 0.25, 0.5, 1.0])
    gripper_values = np.array([[0, 0], [0.9, 0.1], [0.1, 0.9], [0, 0]])
    commands = robotwin.decode_chunk(
        direct_parameters, query, gripper_times, gripper_values
    )
    np.testing.assert_allclose(
        commands[:, robotwin.ARM_CHANNELS], query[:, None] * slopes
    )
    np.testing.assert_array_equal(commands[:, [6, 13]], gripper_values[[1, 2, 2]])


def test_bounded_joint_angles_are_not_unwrapped_as_cartesian_rotations():
    actions = np.zeros((2, 14))
    initial = np.zeros(14)
    initial[0] = 3.0
    actions[:, 0] = [-3.0, 3.0]
    commands, _, _ = robotwin.adapt_chunk(actions, initial, speed_factor=1.5)
    assert commands[0, 0] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n", [1, 2, 7, 16])
@pytest.mark.parametrize("speed", [1.0, 1.5, 2.0])
def test_scaling_preserves_source_span(n, speed):
    t = np.arange(n + 1) * 0.1
    query = robotwin.source_schedule(t, speed)
    assert query[-1] == t[-1]
    assert np.all(np.diff(query) > 0)
    assert query[0] > 0
    assert len(query) == int(np.ceil(n / speed))
    assert query[-1] / speed == pytest.approx(n * 0.1 / speed)


def test_dense_one_returns_original_object():
    actions, initial = joint_chunk()
    decoded, spline, info = robotwin.adapt_chunk(actions, initial)
    assert decoded is actions
    assert spline is None
    assert info["fallback_count"] == 0


@pytest.mark.parametrize("speed", [1, 1.5, 2])
def test_spline_and_dense_share_schedule_and_preserve_gripper_edges(speed):
    actions, initial = joint_chunk()
    # Close->open->close within two samples, then several later transitions.
    actions[:, 6] = [0.9, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1] * 2
    actions[:, 13] = 0.85
    dense, _, di = robotwin.adapt_chunk(actions, initial, speed_factor=speed)
    spline_commands, spline, si = robotwin.adapt_chunk(
        actions, initial, action_adapter="spline", speed_factor=speed
    )
    assert spline is not None and si["fallback_count"] == 0
    np.testing.assert_array_equal(si["query_source_times"], di["query_source_times"])
    np.testing.assert_array_equal(spline_commands[:, [6, 13]], dense[:, [6, 13]])
    assert spline_commands[0, 6] == actions[0, 6]
    close_index = si["query_source_times"].index(0.2)
    assert spline_commands[close_index, 6] == actions[1, 6]
    assert spline.source_domain == (0.0, 1.6)


def test_speed_does_not_change_saved_spline_parameters():
    actions, initial = joint_chunk()
    splines = [
        robotwin.adapt_chunk(actions, initial, action_adapter="spline", speed_factor=s)[
            1
        ]
        for s in [1, 1.5, 2]
    ]
    for spline in splines[1:]:
        np.testing.assert_array_equal(spline.knots, splines[0].knots)
        np.testing.assert_array_equal(spline.control_points, splines[0].control_points)


@pytest.mark.parametrize("speed", [1, 1.5, 2])
def test_invalid_fits_fall_back_to_corresponding_dense_path(monkeypatch, speed):
    actions, initial = joint_chunk()
    dense, _, di = robotwin.adapt_chunk(actions, initial, speed_factor=speed)

    def fail(*args, **kwargs):
        raise RuntimeError("injected fitting failure")

    monkeypatch.setattr(robotwin, "fit_spline", fail)
    commands, _, info = robotwin.adapt_chunk(
        actions, initial, action_adapter="spline", speed_factor=speed
    )
    np.testing.assert_array_equal(commands, dense)
    assert info["query_source_times"] == di["query_source_times"]
    assert info["fallback_count"] == 1
    assert "injected fitting failure" in info["failure"]


def test_intermediate_cubic_overshoot_falls_back():
    actions, initial = joint_chunk(6)
    actions[:, 0] = [0, 1, 0, 1, 0, 1]
    commands, _, info = robotwin.adapt_chunk(
        actions, initial, action_adapter="spline", speed_factor=1.5, smoothing=0
    )
    dense, _, _ = robotwin.adapt_chunk(actions, initial, speed_factor=1.5)
    assert info["fallback_count"] == 1
    assert "Execution-query" in info["failure"]
    np.testing.assert_array_equal(commands, dense)


def test_nonfinite_decode_falls_back_and_remains_loggable(monkeypatch):
    actions, initial = joint_chunk()
    monkeypatch.setattr(
        robotwin,
        "decode_chunk",
        lambda spline, query, *args: np.full((len(query), 14), np.nan),
    )
    decoded, _, info = robotwin.adapt_chunk(
        actions, initial, action_adapter="spline", speed_factor=2
    )
    dense, _, _ = robotwin.adapt_chunk(actions, initial, speed_factor=2)
    assert info["fallback_count"] == 1
    np.testing.assert_array_equal(decoded, dense)
    json.dumps(info, allow_nan=False)


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    float = cpu = detach

    def numpy(self):
        return self.values


class FakeEnv:
    def __init__(self):
        self.calls = []
        self.take_action_cnt = 0
        self.live_sim_step = 100  # exclude setup physics
        self.eval_success = False
        self.scene = SimpleNamespace(get_timestep=lambda: 0.004)

    def get_instruction(self):
        return "test instruction"

    def take_action(self, action, action_type):
        self.calls.append((action.copy(), action_type))
        if self.eval_success:
            return
        self.take_action_cnt += 1
        self.live_sim_step += self.take_action_cnt * 3  # variable command durations
        self.eval_success = self.take_action_cnt == 3


def test_dense_one_matches_untouched_eval_and_episode_state_resets(tmp_path):
    # Execute the actual upstream eval body without importing torch/checkpoints.
    policy_path = (
        Path(__file__).resolve().parents[1]
        / "inference/robotwin/Motus/deploy_policy.py"
    )
    original_tree = ast.parse(policy_path.read_text())
    original_eval = next(
        node
        for node in original_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "eval"
    )
    namespace = {"encode_obs": lambda x: x}
    exec(  # noqa: S102 -- execute only the repository's existing eval body for parity
        compile(
            ast.Module(body=[original_eval], type_ignores=[]), str(policy_path), "exec"
        ),
        namespace,
    )
    actions, initial = joint_chunk()
    model = SimpleNamespace(
        device="cpu",
        set_instruction=lambda x: None,
        update_obs=lambda x: None,
        current_state=FakeTensor(initial),
        get_action=lambda: actions,
    )
    original_env = FakeEnv()
    namespace["eval"](original_env, model, {})
    experiment.configure(model, {"log_dir": str(tmp_path)})
    env = FakeEnv()
    experiment.begin_episode(model, env, {"scene_seed": 10, "task": "test"})
    experiment.eval_chunk(env, model, {})
    assert len(env.calls) == len(original_env.calls) == 16
    for actual, expected in zip(env.calls, original_env.calls):
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[0].dtype == expected[0].dtype
        assert actual[1] == expected[1] == "qpos"
    record = experiment.end_episode(model, env, {"success": True})
    assert record["physics_steps"] == 18
    assert record["simulated_seconds"] == pytest.approx(0.072)
    assert record["chunks"][0]["command_physics_steps"] == [3, 6, 9] + [0] * 13
    assert model.episode_metrics is None and model.previous_command is None
    experiment.begin_episode(model, FakeEnv(), {"scene_seed": 11, "task": "test"})
    assert model.episode_metrics["chunks"] == []
    assert model.episode_metrics["policy_calls"] == 0
    assert model.previous_command is None


def test_policy_seed_resets_each_episode(monkeypatch, tmp_path):
    seed_calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            manual_seed=seed_calls.append,
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    actions, _ = joint_chunk()
    model = SimpleNamespace(get_action=lambda: actions)
    experiment.configure(model, {"log_dir": str(tmp_path), "policy_seed": 42})
    experiment.begin_episode(model, FakeEnv(), {"scene_seed": 10})
    first = (random.random(), np.random.random())
    experiment.begin_episode(model, FakeEnv(), {"scene_seed": 11})
    assert (random.random(), np.random.random()) == first
    assert seed_calls == [42, 42]


def test_failed_inference_latency_is_recorded(tmp_path):
    def fail():
        raise RuntimeError("injected model failure")

    model = SimpleNamespace(device="cpu", get_action=fail)
    experiment.configure(model, {"log_dir": str(tmp_path)})
    env = FakeEnv()
    experiment.begin_episode(model, env, {"scene_seed": 10})
    with pytest.raises(RuntimeError, match="injected model failure"):
        model.get_action()
    record = experiment.end_episode(
        model, env, {"success": False, "failure": "injected model failure"}
    )
    assert record["policy_calls"] == 1
    assert len(record["inference_latencies"]) == 1
    assert record["inference_seconds"] > 0
    assert record["dense_sample_counts"] == []
