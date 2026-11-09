"""Run a small matched RoboTwin matrix, using the existing evaluation harness.

Invoke with the configured runtime Python, inside an existing GPU allocation.
No model implementation is imported by this launcher.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def configurations(
    original=False, adapters=("dense", "spline"), speeds=(1.0, 1.5, 2.0)
):
    return (
        [("original", 1.0)] if original else [(a, s) for s in speeds for a in adapters]
    )


def command(args, adapter, speed, seed, output):
    root = args.robotwin_root.resolve()
    overrides = {
        "policy_name": "Motus",
        "task_name": args.task,
        "task_config": args.task_config,
        "seed": args.policy_seed,
        "policy_seed": args.policy_seed,
        "ckpt_setting": str(args.checkpoint.resolve()),
        "wan_path": str(args.wan_path.resolve()),
        "vlm_path": str(args.vlm_path.resolve()),
        "action_adapter": "dense" if adapter == "original" else adapter,
        "original_execution": adapter == "original",
        "speed_factor": speed,
        "source_dt": args.source_dt,
        "fitting_tolerances": args.tolerances,
        "fitting_smoothing": args.smoothing,
        "max_control_points": args.max_control_points,
        "log_dir": str(output.resolve()),
    }
    cmd = [
        str(args.sim_python),
        str(root / "script/eval_policy.py"),
        "--config",
        str(root / "policy/Motus/deploy_policy.yml"),
        "--overrides",
    ]
    # JSON is valid YAML and preserves list/boolean/null types in the existing parser.
    for key, value in overrides.items():
        cmd.extend(["--" + key, json.dumps(value)])
    env = os.environ.copy()
    env.update(
        MOTUS_ROOT=str(Path(__file__).resolve().parents[2]),
        ROBOTWIN_EVAL_EPISODES="1",
        ROBOTWIN_EVAL_REPLAY_SEED=str(seed),
        # A rejected scene must not silently turn into a different requested seed.
        ROBOTWIN_EVAL_MAX_ATTEMPTS="1",
        ROBOTWIN_EVAL_RESULT_DIR=str(output.resolve()),
        WANDB_MODE="disabled",
    )
    return cmd, env


def preflight(args):
    checkpoint = args.checkpoint.resolve()
    if checkpoint.suffix != ".pt":
        checkpoint = checkpoint / "mp_rank_00_model_states.pt"
    required = {
        "simulator_python": args.sim_python,
        "motus_checkpoint": checkpoint,
        "wan_config": args.wan_path / "config.json",
        "wan_vae": args.wan_path / "Wan2.2_VAE.pth",
        "t5_weights": args.wan_path / "models_t5_umt5-xxl-enc-bf16.pth",
        "t5_tokenizer": args.wan_path / "google/umt5-xxl",
        "qwen_processor": args.vlm_path / "preprocessor_config.json",
        "robot_embodiment": args.robotwin_root
        / "assets/embodiments/aloha-agilex/config.yml",
        "objects": args.robotwin_root / "assets/objects",
    }
    return {
        name: {"path": str(path.absolute()), "exists": path.exists()}
        for name, path in required.items()
    }


def main():
    parent = Path(__file__).resolve().parents[3]
    env_root = Path(os.environ.get("ROBOTWIN_UV_ENV_ROOT", parent / "uv_envs"))
    simulator_env = Path(
        os.environ.get("ROBOTWIN_CLASSIC_ENV", env_root / "robotwin-classic")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, default=parent / "WAM-RoboTwin")
    parser.add_argument("--sim-python", type=Path, default=simulator_env / "bin/python")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wan-path", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--task", default="click_alarmclock")
    parser.add_argument("--task-config", default="demo_randomized")
    parser.add_argument(
        "--scene-seeds", type=int, nargs="+", default=[4300000, 4300001, 4300002]
    )
    parser.add_argument("--policy-seed", type=int, default=42)
    parser.add_argument(
        "--adapters",
        choices=["dense", "spline"],
        nargs="+",
        default=["dense", "spline"],
    )
    parser.add_argument(
        "--speeds",
        type=float,
        choices=[1.0, 1.5, 2.0],
        nargs="+",
        default=[1.0, 1.5, 2.0],
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="Run the untouched upstream Motus eval function",
    )
    parser.add_argument("--source-dt", type=float, default=0.1)
    parser.add_argument("--tolerances", type=float, nargs=12, default=[0.01] * 12)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--max-control-points", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Save prerequisite checks and stop, without loading a model",
    )
    args = parser.parse_args()
    args.sim_python = args.sim_python.absolute()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = preflight(args)
    (args.output / "preflight.json").write_text(json.dumps(checks, indent=2) + "\n")
    if args.preflight:
        print(json.dumps(checks, indent=2))
        return int(any(not check["exists"] for check in checks.values()))
    if not args.dry_run:
        missing = [name for name, check in checks.items() if not check["exists"]]
        if missing:
            parser.error("Missing prerequisites: " + ", ".join(missing))
        if shutil.which("srun") and not os.environ.get("SLURM_JOB_ID"):
            parser.error("Run GPU evaluation inside a Slurm allocation")
    failures = 0
    for adapter, speed in configurations(args.original, args.adapters, args.speeds):
        for seed in args.scene_seeds:
            output = args.output / f"{adapter}_{speed:g}x" / f"seed{seed}"
            cmd, env = command(args, adapter, speed, seed, output)
            print(f"ROBOTWIN_EVAL_REPLAY_SEED={seed} " + shlex.join(cmd), flush=True)
            if args.dry_run:
                continue
            # Existing harness resumes its own journal; avoid duplicate experiment logs.
            if (output / "episodes.jsonl").exists():
                parser.error(
                    f"Output already exists: {output}; choose a fresh --output"
                )
            output.mkdir(parents=True, exist_ok=True)
            (output / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
            with (output / "console.log").open("w") as log:
                result = subprocess.run(
                    cmd,
                    cwd=args.robotwin_root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            (output / "process.json").write_text(
                json.dumps({"exit_code": result.returncode}) + "\n"
            )
            failures += result.returncode != 0
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
