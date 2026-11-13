"""Convert one episode to an immutable, episode-linked spline sidecar."""
import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_spline import (CHANNELS, VERSION, decode, fit_episode, identity,
    load_actions, normalize, range_diagnostics, source_identity)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode-dir", type=Path, required=True, help="One task/config data/ or qpos/ directory")
    p.add_argument("--episode", type=Path, help="Default: first training episode in seeded split")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--normalization", type=Path, help="Previously generated normalization JSON; reuse unchanged")
    p.add_argument("--diagnostic-only", action="store_true", help="Explicitly allow single-episode provisional bounds")
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--max-control-points", type=int)
    p.add_argument("--seconds-per-step", type=float, help="Only supply a verified recording interval")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)
    if not 0 < args.validation_fraction < 1:
        p.error("validation fraction must lie in (0,1)")
    if args.seconds_per_step is not None and (not np.isfinite(args.seconds_per_step) or args.seconds_per_step <= 0):
        p.error("seconds per step must be finite and positive")
    root = args.episode_dir.resolve(strict=True)
    paths = sorted(f.resolve() for f in root.iterdir() if f.suffix in (".hdf5", ".h5", ".pt", ".npy"))
    if not paths:
        p.error(f"No episodes in {root}; no data was downloaded")
    output = args.output.resolve()
    norm_path = output.with_suffix(".normalization.json")
    if output.exists() or (not args.normalization and norm_path.exists()):
        p.error("Output exists; choose a new sidecar path")
    if output in paths or norm_path in paths:
        p.error("Output cannot replace source data")
    if args.normalization:
        norm = json.loads(args.normalization.read_text())
        claimed = norm.pop("identity")
        if claimed != identity(norm):
            p.error("Normalization identity mismatch")
        norm["identity"] = claimed
        if norm["scope"] == "diagnostic-only" and not args.diagnostic_only:
            p.error("Diagnostic normalization requires --diagnostic-only")
        if norm["scope"] not in ("training-only", "diagnostic-only"):
            p.error("Unknown normalization scope")
        episode = (args.episode or paths[0]).resolve(strict=True)
    else:
        order = np.random.default_rng(args.seed).permutation(len(paths))
        if len(paths) == 1:
            if not args.diagnostic_only:
                p.error("Only one episode: explicitly pass --diagnostic-only; unsuitable for training statistics")
            train, validation, scope = paths, [], "diagnostic-only"
        else:
            nval = min(len(paths) - 1, max(1, int(np.ceil(len(paths) * args.validation_fraction))))
            validation = [paths[i] for i in order[:nval]]
            train = [paths[i] for i in order[nval:]]
            scope = "training-only"
        lows, highs, sources = [], [], []
        # Split is fixed before reading any action values. Stream one episode at a time.
        for path in train:
            a = load_actions(path)
            lows.append(a.min(axis=0))
            highs.append(a.max(axis=0))
            sources.append(source_identity(path, a))
        norm = dict(schema=VERSION, scope=scope, transform="affine_minmax_to_minus1_plus1_no_clip",
                    low=np.min(lows, axis=0).tolist(), high=np.max(highs, axis=0).tolist(),
                    training_sources=sources, validation_paths=[str(f) for f in validation],
                    seed=args.seed, validation_fraction=args.validation_fraction)
        norm["identity"] = identity(norm)
        episode = (args.episode or train[0]).resolve(strict=True)
    if episode not in paths:
        p.error("Episode must belong to --episode-dir")
    actions = load_actions(episode)
    low, high = np.asarray(norm["low"]), np.asarray(norm["high"])
    if low.shape != (14,) or high.shape != (14,) or not np.isfinite([low, high]).all() or np.any(high < low):
        p.error("Invalid normalization bounds")
    y = normalize(actions, norm)
    spline, diagnostics = fit_episode(y, args.tolerance, args.max_control_points)
    diagnostics["ranges"] = range_diagnostics(spline, actions, norm)
    diagnostics["per_channel_max_abs_decoded_residual"] = np.abs(decode(spline(np.arange(len(actions))), norm) - actions).max(axis=0).tolist()
    source = source_identity(episode, actions)
    metadata = dict(schema=VERSION, task="place_can_basket", embodiment="aloha_agilex",
        channels=CHANNELS, channel_units=["radian"] * 6 + ["unitless_open_fraction"] + ["radian"] * 6 + ["unitless_open_fraction"],
        normalization=norm, source=source, diagnostics=diagnostics,
        time=dict(units="source_action_step", origin=0, step=1, seconds_per_step=args.seconds_per_step,
                  seconds_mapping_status="user_verified" if args.seconds_per_step else "unknown",
                  camera_mapping="same recorded row; physical timestamps unverified"),
        images=dict(source_episode=str(episode), copied=False,
                    preprocessed_video=str(episode.parent.parent / "videos" / (episode.stem + ".mp4")) if episode.suffix == ".pt" else None),
        converter_sha256=identity({f.name: f.read_text() for f in [Path(__file__), Path(__file__).with_name("adaptive_spline.py")]}))
    metadata["cache_identity"] = identity(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.normalization:
        with norm_path.open("x") as f:
            json.dump(norm, f, indent=2, allow_nan=False)
    with output.open("xb") as f:
        np.savez_compressed(f, knots=spline.t, control_points=spline.c, degree=spline.k,
                            actions=actions, source_indices=np.arange(len(actions)),
                            metadata=json.dumps(metadata, allow_nan=False))
    print(json.dumps(dict(output=str(output), success=diagnostics["success"], reason=diagnostics["reason"],
        samples=len(actions), control_points=len(spline.c), normalization_scope=norm["scope"],
        max_abs_normalized_residual=diagnostics["max_abs_normalized_residual"]), indent=2))
    if args.plot:
        from visualize_episode_spline import visualize
        visualize(output)
    return 0 if diagnostics["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
