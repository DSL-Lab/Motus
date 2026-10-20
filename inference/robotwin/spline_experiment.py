"""Phase 1 preflight, original rollout, opt-in reconstruction, and reporting.

Run with -m from the Motus checkout. No training entry point is provided.
"""

import argparse
import hashlib
import importlib
import json
import os
import random
import runpy
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/spline/phase1.json"
POLICY = ROOT / "inference/robotwin/Motus"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def revision(directory):
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()


def source_digest():
    digest = hashlib.sha256()
    files = [CONFIG, *Path(__file__).parent.glob("spline_*.py")]
    files += [p for p in POLICY.rglob("*") if p.suffix in (".py", ".yml", ".json")]
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def preflight(args):
    repo = args.robotwin.resolve()
    env_root = Path(os.environ.get("ROBOTWIN_UV_ENV_ROOT", ROOT.parent / "uv_envs"))
    classic = Path(
        os.environ.get("ROBOTWIN_CLASSIC_ENV", env_root / "robotwin-classic")
    )
    interpreter = classic / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    errors = []
    checks = {
        "robotwin_eval": repo / "script/eval_policy.py",
        "shared_interpreter": interpreter,
        "task_config_directory": repo / "task_config",
        "checkpoint": (
            args.checkpoint / "mp_rank_00_model_states.pt" if args.checkpoint else None
        ),
        "wan_config": (args.wan / "config.json" if args.wan else None),
        "wan_vae": (args.wan / "Wan2.2_VAE.pth" if args.wan else None),
        "wan_t5": (args.wan / "models_t5_umt5-xxl-enc-bf16.pth" if args.wan else None),
        "wan_tokenizer": (args.wan / "google/umt5-xxl" if args.wan else None),
        "vlm": args.vlm,
    }
    for name, path in checks.items():
        if path is None or not path.exists():
            errors.append(f"Missing {name}: {path}")
    if (
        checks["task_config_directory"].exists()
        and not checks["task_config_directory"].is_dir()
    ):
        errors.append(
            "task_config is not a directory: restore Git symlinks in a Linux checkout"
        )
    return {
        "status": "blocked" if errors else "paths_ready",
        "baseline_rollout_completed": False,
        "motus_revision": revision(ROOT),
        "robotwin_revision": revision(repo),
        "paths": {
            k: str(v.resolve()) if v is not None else None for k, v in checks.items()
        },
        "errors": errors,
        "runtime_unverified": [
            "CUDA and VRAM",
            "Motus dependencies in selected shared environment",
            "simulator assets/rendering",
            "checkpoint compatibility",
        ],
        "timing_constraint": "TOPP target duration varies; speed_factor > 1 cannot use a fixed execution clock",
    }


def load_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run(args, config):
    report = preflight(args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight.json", report)
    if report["errors"]:
        raise RuntimeError("Preflight blocked; see " + str(output / "preflight.json"))
    if (
        Path(sys.executable).resolve()
        != Path(report["paths"]["shared_interpreter"]).resolve()
    ):
        raise RuntimeError(
            "Run with the RoboTwin shared interpreter shown in preflight.json"
        )
    if args.speed_factor != 1.0:
        raise RuntimeError(
            "Accelerated execution blocked: take_action(qpos) runs variable-duration TOPP. "
            "The adapter's fixed-rate time map cannot be honored by this executor. "
            "Do not substitute waypoint skipping or change controllers silently."
        )
    identity = {
        "inference_source_sha256": source_digest(),
        "motus_revision": report["motus_revision"],
        "robotwin_revision": report["robotwin_revision"],
        "task": args.task,
        "split": args.split,
        "seed": args.seed,
        "episodes": args.episodes,
        "instruction_type": config["instruction_type"],
        "paths": report["paths"],
        "checkpoint_bytes": Path(report["paths"]["checkpoint"]).stat().st_size,
        "checkpoint_mtime_ns": Path(report["paths"]["checkpoint"]).stat().st_mtime_ns,
    }
    baseline = None
    if args.mode != "original":
        if args.baseline is None:
            raise RuntimeError(
                "Run the original policy first and pass its completed --baseline directory"
            )
        baseline = json.loads(
            (args.baseline / "complete.json").read_text(encoding="utf-8")
        )
        if baseline["mode"] != "original" or baseline["identity"] != identity:
            raise RuntimeError(
                "Baseline must match source revisions, paths, task, split, seed, and episode count"
            )
    if (output / "run.json").exists() or (output / "_episodes.jsonl").exists():
        raise RuntimeError(
            "Choose a fresh output directory; partial runs are not silently resumed"
        )
    write_json(
        output / "run.json",
        {
            "identity": identity,
            "mode": args.mode,
            "speed_factor": args.speed_factor,
            "configuration": config,
        },
    )
    # Import simulator/model dependencies only after paths and baseline gate pass.
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "No allocated CUDA device; original Motus rollout cannot start"
        )
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ["ROBOTWIN_EVAL_EPISODES"] = str(args.episodes)
    os.environ["ROBOTWIN_EVAL_RESULT_DIR"] = str(output)
    repo = args.robotwin.resolve()
    # Reuse this sibling fork's existing legacy evaluator; do not copy a policy
    # or add anything to RoboTwin. Its policy decorator imports the name Motus.
    sys.path[:0] = [str(POLICY.parent), str(POLICY), str(repo / "script"), str(repo)]
    os.chdir(repo)
    policy = importlib.import_module("Motus")
    from inference.robotwin.spline_rollout import RolloutRecorder, install_hooks

    recorder = RolloutRecorder(
        policy,
        output,
        args.mode,
        args.speed_factor,
        config["joint_tolerances_rad"],
        config["max_controls"],
    )
    entry = runpy.run_path(str(repo / "script/eval_policy.py"))
    namespace = entry["main"].__globals__
    install_hooks(namespace, policy, recorder)
    # Same render smoke check as the evaluator's normal __main__ entry point.
    from test_render import Sapien_TEST

    Sapien_TEST()
    namespace["main"](
        {
            "policy_name": "Motus",
            "task_name": args.task,
            "task_config": args.split,
            "ckpt_setting": str(Path(report["paths"]["checkpoint"]).parent),
            "seed": args.seed,
            "instruction_type": config["instruction_type"],
            "log_dir": str(output),
            "wan_path": str(Path(report["paths"]["wan_config"]).parent),
            "vlm_path": report["paths"]["vlm"],
        }
    )
    episodes = load_jsonl(output / "_episodes.jsonl")
    if len(episodes) != args.episodes or len(recorder.episodes) != args.episodes:
        raise RuntimeError("Missing episode journals or timing records")
    scenes = [[e["seed"], e["instruction"]] for e in episodes]
    if baseline is not None and scenes != baseline["scenes"]:
        raise RuntimeError(
            "Scene/instruction mismatch against original baseline; comparison invalid"
        )
    write_json(
        output / "complete.json",
        {
            "identity": identity,
            "mode": args.mode,
            "speed_factor": args.speed_factor,
            "scenes": scenes,
        },
    )


def summarize(args):
    import statistics

    rows = []
    for complete in sorted(args.output.rglob("complete.json")):
        metadata = json.loads(complete.read_text(encoding="utf-8"))
        directory = complete.parent
        episodes = load_jsonl(directory / "_episodes.jsonl")
        timing = load_jsonl(directory / "timing.jsonl")
        chunks = load_jsonl(directory / "chunks.jsonl")
        successful = [t for t, e in zip(timing, episodes) if e["success"]]
        latency = [
            c["observation_to_action_s"]
            for c in chunks
            if c.get("observation_to_action_s") is not None
        ]
        mean = lambda values: statistics.mean(values) if values else None
        joint_errors = [
            c["max_error_per_joint"] for c in chunks if "max_error_per_joint" in c
        ]
        max_errors = [
            max((e[j] for e in joint_errors if e[j] is not None), default=None)
            for j in range(12)
        ]
        rows.append(
            {
                "run": str(directory),
                "task": metadata["identity"]["task"],
                "split": metadata["identity"]["split"],
                "mode": metadata["mode"],
                "speed_factor": metadata["speed_factor"],
                "episodes": len(episodes),
                "success_rate": mean([int(e["success"]) for e in episodes]),
                "successful_completion_simulated_s": mean(
                    [t["simulated_s"] for t in successful]
                ),
                "successful_completion_wall_s": mean([t["wall_s"] for t in successful]),
                "all_episode_simulated_s": mean([t["simulated_s"] for t in timing]),
                "all_episode_wall_s": mean([t["wall_s"] for t in timing]),
                "policy_calls_mean": mean([t["policy_calls"] for t in timing]),
                "observation_to_action_mean_s": mean(latency),
                "observation_to_action_max_s": max(latency, default=None),
                "processing_per_chunk_mean_s": mean(
                    [c["processing_s"] for c in chunks]
                ),
                "fallback_frequency": mean([int(c["fallback"]) for c in chunks]),
                "max_reconstruction_error_per_joint": max_errors,
                "debug_artifact_bytes": sum(c["artifact_bytes"] for c in chunks),
                "dense_payload_bytes": sum(c["dense_bytes"] for c in chunks),
                "representation_payload_bytes": sum(
                    c["representation_bytes"] for c in chunks
                ),
                "failure_artifacts": [
                    {"artifact": c["artifact"], "reason": c["fit_failure"]}
                    for c in chunks
                    if c["fallback"]
                ],
            }
        )
    write_json(
        args.output / "summary.json",
        {
            "rollouts": rows,
            "phase1_complete": False,
            "training_gate": "closed: full matched six-variant experiment required",
        },
    )
    print(json.dumps(rows, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "summarize", "plan"))
    parser.add_argument("--robotwin", type=Path, default=ROOT.parent / "WAM-RoboTwin")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wan", type=Path)
    parser.add_argument("--vlm", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "logs/spline")
    parser.add_argument(
        "--mode", choices=("original", "dense", "spline"), default="original"
    )
    parser.add_argument(
        "--speed-factor", type=float, choices=(1.0, 1.5, 2.0), default=1.0
    )
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument(
        "--split", choices=("demo_clean", "demo_randomized"), default="demo_clean"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    try:
        if args.command == "check":
            result = preflight(args)
            write_json(args.output / "preflight.json", result)
            print(json.dumps(result, indent=2))
            return int(bool(result["errors"]))
        if args.command == "run":
            run(args, config)
        elif args.command == "summarize":
            summarize(args)
        else:
            tasks = (
                (ROOT / config["tasks_file"])
                .resolve()
                .read_text(encoding="utf-8")
                .splitlines()
            )
            print(
                json.dumps(
                    {
                        "tasks": tasks,
                        "config": config,
                        "first_step": "original rollout on first task, one episode",
                    },
                    indent=2,
                )
            )
    except Exception as error:  # noqa: BLE001 -- journal any simulator/model failure before exiting
        write_json(
            args.output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "rollout_completed": False,
            },
        )
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
