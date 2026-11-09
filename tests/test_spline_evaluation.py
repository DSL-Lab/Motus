import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "experiments/spline" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_matched_matrix_commands_keep_seeds_settings_and_typed_overrides(tmp_path):
    launcher = load_script("evaluate")
    args = SimpleNamespace(
        robotwin_root=ROOT.parent / "WAM-RoboTwin",
        sim_python=Path("/existing/runtime/python"),
        task="click_alarmclock",
        task_config="demo_randomized",
        policy_seed=42,
        checkpoint=tmp_path / "checkpoint with spaces",
        wan_path=tmp_path / "wan",
        vlm_path=tmp_path / "qwen",
        source_dt=0.1,
        tolerances=[0.01] * 12,
        smoothing=1.0,
        max_control_points=None,
    )
    configs = launcher.configurations()
    assert len(configs) == 6 and len(set(configs)) == 6
    for adapter, speed in configs:
        cmd, env = launcher.command(args, adapter, speed, 4300000, tmp_path / "results")
        pairs = cmd[cmd.index("--overrides") + 1 :]
        overrides = {
            pairs[i]: json.loads(pairs[i + 1]) for i in range(0, len(pairs), 2)
        }
        assert overrides["--ckpt_setting"] == str(args.checkpoint)
        assert overrides["--action_adapter"] == adapter
        assert overrides["--speed_factor"] == speed
        assert overrides["--policy_seed"] == 42
        assert overrides["--fitting_tolerances"] == [0.01] * 12
        assert overrides["--max_control_points"] is None
        assert env["ROBOTWIN_EVAL_REPLAY_SEED"] == "4300000"
        assert env["ROBOTWIN_EVAL_EPISODES"] == env["ROBOTWIN_EVAL_MAX_ATTEMPTS"] == "1"
    assert launcher.configurations(original=True) == [("original", 1.0)]


def test_summary_uses_successful_times_and_only_matched_successful_pairs(tmp_path):
    reporter = load_script("summarize")
    for adapter, speed, times in [
        ("dense", 1, [10, 20, 30]),
        ("spline", 2, [4, 9, 99]),
    ]:
        for seed, seconds in enumerate(times):
            folder = tmp_path / f"{adapter}_{speed}x" / f"seed{seed}"
            folder.mkdir(parents=True)
            record = {
                "task": "test",
                "task_config": "demo_clean",
                "scene_seed": seed,
                "policy_seed": 42,
                "instruction": "same instruction",
                "configuration": {"ckpt_setting": "same checkpoint"},
                "action_options": {"action_adapter": adapter, "speed_factor": speed},
                "success": seed < 2,
                "simulated_seconds": seconds,
                "wall_seconds": 100 + seconds,
                "fallback_count": 0,
            }
            # A different instruction must not count as a matched speedup pair.
            if adapter == "spline" and seed == 1:
                record["instruction"] = "different instruction"
            (folder / "episodes.jsonl").write_text(json.dumps(record) + "\n")
    rows = reporter.summarize(tmp_path)
    baseline, spline = rows
    assert baseline["success_rate"] == spline["success_rate"] == 2 / 3
    assert baseline["successful_mean_simulated_seconds"] == 15
    assert spline["successful_mean_simulated_seconds"] == 6.5
    assert spline["matched_successful_pairs"] == 1
    assert spline["achieved_mean_paired_speedup"] == 2.5


def test_empty_report_does_not_manufacture_results(tmp_path):
    assert load_script("summarize").summarize(tmp_path) == []
