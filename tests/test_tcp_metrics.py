"""Reconstruction diagnostics must not affect policy execution or fallback."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from motus_actions import experiment
from motus_actions.robotwin import adapt_chunk
from motus_actions.tcp_metrics import (
    chunk_position_metrics,
    pool_statistics,
    position_statistics,
)


class CartesianFK:
    """Analytic fixture with an independent, explicit position for each arm."""

    def __init__(self, _robot=None):
        self.metadata = {"backend": "analytic test fixture"}

    def verify_live_tcp(self, _robot):
        return {"left": 0.0, "right": 0.0}

    def positions(self, values):
        values = np.asarray(values)
        return np.stack([values[:, :3], values[:, 6:9]], axis=1)


def test_euclidean_position_statistics_pool_squared_samples():
    reference = np.zeros((1, 2, 3))
    candidate = np.array([[[0.003, 0.004, 0], [0, 0, 0.012]]])
    errors = position_statistics(candidate, reference)
    assert errors["left"]["position_rmse_mm"] == 5
    assert errors["right"]["max_position_error_mm"] == 12
    zeros = position_statistics(np.zeros((8, 2, 3)), np.zeros((8, 2, 3)))
    pooled = pool_statistics([errors, zeros])
    assert pooled["left"]["sample_count"] == 9
    assert pooled["left"]["position_rmse_mm"] == pytest.approx(5 / 3)
    assert pooled["left"]["max_position_error_mm"] == 5
    empty = pool_statistics([])
    assert empty["right"]["sample_count"] == 0
    assert empty["right"]["position_rmse_mm"] is None


@pytest.mark.parametrize("fail_metrics", [False, True])
def test_future_logging_preserves_rejected_candidate_fallback_and_noops(
    monkeypatch, tmp_path, fail_metrics
):
    initial = np.zeros(14)
    actions = np.zeros((16, 14), dtype=np.float32)
    actions[:, 0] = 0.2 * np.sin(np.arange(1, 17) * 1.7)
    options = {"action_adapter": "spline", "speed_factor": 1.5}
    expected, spline, info = adapt_chunk(actions, initial, **options)
    assert info["fallback_count"] == 1 and spline is not None
    tensor = SimpleNamespace(numpy=lambda: initial)
    tensor.detach = tensor.float = tensor.cpu = lambda: tensor
    model = SimpleNamespace(
        device="cpu",
        current_state=tensor,
        get_action=lambda: actions,
        set_instruction=lambda _: None,
        update_obs=lambda _: None,
    )
    experiment.configure(model, {"log_dir": str(tmp_path), **options})
    calls = []
    env = SimpleNamespace(
        robot=object(),
        live_sim_step=100,
        take_action_cnt=0,
        scene=SimpleNamespace(get_timestep=lambda: 0.004),
        get_instruction=lambda: "fixture",
    )

    def take_action(action, action_type):
        calls.append((action.copy(), action_type))
        if len(calls) <= 3:
            env.live_sim_step += 4
            env.take_action_cnt += 1

    env.take_action = take_action
    monkeypatch.setattr(experiment, "RobotTCPKinematics", CartesianFK)
    experiment.begin_episode(model, env, {"scene_seed": 42})
    if fail_metrics:

        def fail(*args):
            raise ValueError("diagnostic failed")

        monkeypatch.setattr(experiment, "chunk_position_metrics", fail)
    experiment.eval_chunk(env, model, {})
    record = experiment.end_episode(model, env, {"success": True})
    np.testing.assert_array_equal(np.array([c[0] for c in calls]), expected)
    assert all(c[1] == "qpos" for c in calls)
    assert record["physics_steps"] == 12
    assert record["fallback_count"] == 1 and record["success"]
    chunk = record["chunks"][0]
    assert chunk["command_physics_steps"] == [4] * 3 + [0] * (len(expected) - 3)
    if fail_metrics:
        assert "diagnostic failed" in chunk["tcp_metrics_unavailable"]
        assert record["tcp_position_metrics"]["measured_chunks"] == 0
    else:
        tcp = chunk["tcp_position_metrics"]
        assert tcp["candidate_status"] == "rejected"
        assert tcp["candidate_source"]["left"]["sample_count"] == 17
        assert tcp["candidate_query"]["left"]["max_position_error_mm"] > 0
        assert tcp["dispatched_query"]["left"]["position_rmse_mm"] == 0
        assert tcp["dispatched_query_physics_positive"]["right"]["sample_count"] == 3
        assert tcp["zero_physics_query_indices"] == list(range(3, len(expected)))
        assert (
            record["tcp_position_metrics"]["candidates"]["rejected"]["candidate_query"][
                "left"
            ]
            == tcp["candidate_query"]["left"]
        )
    saved = json.loads((tmp_path / "episodes.jsonl").read_text())
    assert saved == record


def test_identical_trajectory_is_zero_and_absent_candidate_is_unavailable():
    actions, initial = np.zeros((8, 14)), np.zeros(14)
    commands, spline, info = adapt_chunk(
        actions, initial, action_adapter="spline", smoothing=0
    )
    metrics = chunk_position_metrics(
        CartesianFK(),
        actions,
        initial,
        info["source_times"],
        info["query_source_times"],
        commands,
        spline,
        info,
    )
    for scope in ("candidate_source", "candidate_query", "dispatched_query"):
        for arm in ("left", "right"):
            assert (
                metrics[scope][arm]["position_rmse_mm"]
                == metrics[scope][arm]["max_position_error_mm"]
                == 0
            )
    metrics = chunk_position_metrics(
        CartesianFK(),
        actions,
        initial,
        info["source_times"],
        info["query_source_times"],
        commands,
        None,
        {"fallback_count": 1, "failure": "fit rejected"},
    )
    assert metrics["candidate_status"] == "rejected"
    assert metrics["candidate_source"] is metrics["candidate_query"] is None
    assert metrics["candidate_unavailable_reason"] == "fit rejected"
    assert not metrics["tracking"]["available"]
