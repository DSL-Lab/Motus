"""Offline RoboTwin cache preparation and one inspectable training example.

Run with python -m motus_actions.prepare in a Slurm allocation. Raw HDF5 is read-only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .spline import AffineScale, VERSION, decode, extract_local, fit_episode, video_indices


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_episode(path):
    import h5py
    with h5py.File(path, "r") as f:
        actions = f["action/joint_states"][:]
        states = f["state/joint_states"][:]
        instruction = f["instruction"][()]
    if isinstance(instruction, bytes):
        instruction = instruction.decode()
    if actions.shape != states.shape or actions.shape[1] != 14:
        raise ValueError(f"invalid 14-D episode: {path}")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(states)):
        raise ValueError(f"nonfinite episode: {path}")
    return actions, states, str(instruction)


def read_frames(path, indices, video_size=(384, 320)):
    """Read only selected JPEGs. Same head/top, wrists/bottom layout as MOTUS."""
    import cv2
    import h5py
    from data.utils.multi_camera_concat import resize_and_concatenate_frames
    reference_root = str(Path(__file__).resolve().parents[2] / 'WAM-RoboTwin')
    if reference_root not in sys.path:
        sys.path.insert(0, reference_root)
    from XPolicyLab.utils.process_data import decode_image_bit
    frames = []
    with h5py.File(path, "r") as f:
        for index in indices:
            cameras = []
            for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
                data = f[f"vision/{camera}/colors"][int(index)]
                frame = decode_image_bit(data)
                if frame is None:
                    raise ValueError(f"invalid JPEG {path}:{camera}:{index}")
                cameras.append(frame)
            combined = resize_and_concatenate_frames(None, *cameras)
            if combined is None:
                raise ValueError("camera layout failed")
            combined = cv2.resize(combined, (video_size[1], video_size[0]))
            frames.append(combined)
    return np.stack(frames)


def select_episode_files(directory,episodes_per_task):
    """Select distinct recorded episodes; never quietly turn a 500-demo run into 50."""
    if isinstance(episodes_per_task,bool) or not isinstance(episodes_per_task,int) or episodes_per_task<2:
        raise ValueError('episodes_per_task must be an integer >=2')
    files=sorted(directory.glob('episode_*.hdf5'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
    if len(files)<episodes_per_task:
        raise ValueError(f'requested {episodes_per_task} demonstrations, found {len(files)} in {directory}')
    return files[:episodes_per_task]


def prepare(raw_root, output, tasks, episodes_per_task=50):
    selected={task:select_episode_files(raw_root/task/'aloha_agilex/data',episodes_per_task) for task in tasks}
    output.mkdir(parents=True, exist_ok=False)
    episodes, split = [], {"train": [], "validation": []}
    for task in tasks:
        files = selected[task]
        # Deterministic episode-level split before fitting any statistics.
        order = np.random.default_rng(0).permutation(len(files))
        val_ids = set(order[:max(1, len(files) // 10)].tolist())
        for i, path in enumerate(files):
            actions, states, instruction = load_episode(path)
            entry = {"id": f"{task}/{path.stem}", "task_name": task,
                     "episode_name": path.stem, "hdf5_path": str(path.resolve()),
                     "sha256": digest(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                     "length": len(actions), "instruction": instruction}
            episodes.append((entry, actions, states))
            split["validation" if i in val_ids else "train"].append(entry["id"])
    train_actions = np.concatenate([a for e, a, _ in episodes if e["id"] in split["train"]])
    action_scale = AffineScale(train_actions.min(0), train_actions.max(0))
    low = np.full(15, np.inf)
    high = np.full(15, -np.inf)
    manifest = {"version": VERSION, "converter_sha256": digest(__file__),
                "spline_sha256": digest(Path(__file__).with_name("spline.py")),
                "split_seed": 0, "tolerance": 0.01, "rows": 36, "video_stride": 4,
                "source_time_units": "sample_index", "split": split,"episodes_per_task":episodes_per_task,
                "action_low": action_scale.low.tolist(), "action_high": action_scale.high.tolist(),
                "episodes": [], "excluded": {"video_padding": 0, "coverage": 0}}
    for entry, actions, states in episodes:
        curve, error = fit_episode(action_scale.encode(actions))
        windows, indices, observations = [], [], []
        for observation in range(max(0, len(actions) - 48)):
            local = extract_local(curve, observation)
            try:
                targets = video_indices(local, observation, len(actions))
            except ValueError:
                manifest["excluded"]["video_padding"] += 1
                continue
            try:
                decode(local, np.arange(1, 17) * 3)
            except ValueError:
                manifest["excluded"]["coverage"] += 1
                continue
            windows.append(local)
            indices.append(targets)
            observations.append(observation)
        if not windows:
            raise ValueError(f"no eligible windows for {entry['id']}")
        windows = np.asarray(windows)
        if entry["id"] in split["train"]:
            low = np.minimum(low, windows.min(axis=(0, 1)))
            high = np.maximum(high, windows.max(axis=(0, 1)))
        path = output / f"{entry['id']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, actions=actions, states=states, knots=curve.t, controls=curve.c,
                 windows=windows.astype(np.float32), video_indices=np.asarray(indices),
                 observations=np.asarray(observations))
        entry.update(cache_path=str(path.resolve()), fit_max_error=error,
                     windows=len(windows), knots=len(curve.t))
        manifest["episodes"].append(entry)
        print(entry["id"], "knots", len(curve.t), "windows", len(windows), "error", error, flush=True)
    manifest.update(parameter_low=low.tolist(), parameter_high=high.tolist())
    manifest["identity"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    example(manifest, output)
    return manifest


def example(manifest, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    entry = next(e for e in manifest["episodes"] if e["id"] in manifest["split"]["train"])
    with np.load(entry["cache_path"]) as data:
        index = len(data["observations"]) // 2
        observation = int(data["observations"][index])
        local = data["windows"][index]
        targets = data["video_indices"][index]
        times = np.arange(1, 17) * 3
        action_scale = AffineScale(manifest["action_low"], manifest["action_high"])
        parameter_scale = AffineScale(manifest["parameter_low"], manifest["parameter_high"])
        decoded = action_scale.decode(decode(local, times))
        frames = read_frames(entry["hdf5_path"], np.r_[observation, targets])
        np.savez(output / "example.npz", current_index=observation, current_image=frames[0],
                 source_actions=data["actions"], global_knots=data["knots"], global_controls=data["controls"],
                 local=local, packed=parameter_scale.encode(local), parameter_mask=np.ones_like(local),
                 video_indices=targets, video_images=frames[1:], valid_domain=local[[3, -4], 0],
                 controller_offsets=times, decoded_actions=decoded)
        figure, axes = plt.subplots(3, 1, figsize=(14, 9))
        for channel in (0, 3, 6, 7, 10, 13):
            axes[0].plot(data["actions"][:, channel], label=f"raw {channel}", alpha=.5)
            axes[0].plot(observation + times, decoded[:, channel], ".")
        for t in data["knots"]:
            axes[0].axvline(t, color="gray", alpha=.1)
        for t in targets:
            axes[0].axvline(t, color="red", alpha=.4)
        axes[0].axvline(observation, color="black")
        axes[0].set_xlabel("Source sample index (physical timestamps unavailable)")
        axes[0].legend(ncol=6)
        axes[1].plot(local[:, 0], ".-")
        axes[1].set_ylabel("Local knot time")
        axes[2].imshow(np.concatenate(list(frames), axis=1))
        axes[2].set_xticks([])
        axes[2].set_yticks([])
        axes[2].set_xlabel(f"Current {observation}; future {targets.tolist()}")
        figure.tight_layout()
        figure.savefig(output / "example.png")
        plt.close(figure)


def prepare_language(source, output, wan_path):
    """Reuse numerically verified spline caches and encode the checkpoint prompt.

    Writes a new manifest/language directory; never rewrites old caches or T5
    embeddings. Rechecking all windows makes the source-hash transition explicit.
    """
    import torch
    from scipy.interpolate import BSpline
    from data.robotwin2.robotwin_data_convert.robotwin_converter import T5EmbeddingProcessor
    from utils.robotwin_prompt import CONDITIONING, robotwin_prompt
    manifest = json.loads(source.read_text())
    output.mkdir(parents=True, exist_ok=False)
    worst = 0.
    for entry in manifest['episodes']:
        raw = Path(entry['hdf5_path']).stat()
        if raw.st_size != entry['bytes'] or raw.st_mtime_ns != entry['mtime_ns']:
            raise ValueError('raw dataset identity changed')
        with np.load(entry['cache_path']) as cached:
            global_curve = BSpline(cached['knots'], cached['controls'], 3, extrapolate=False)
            for current, local in zip(cached['observations'], cached['windows']):
                offsets = np.arange(1, 17) * 3
                error = np.max(np.abs(decode(local, offsets) - global_curve(current + offsets)))
                worst = max(worst, float(error))
    if worst > 1e-5:
        raise ValueError(f'reused local/global mismatch {worst}')
    processor = T5EmbeddingProcessor(str(wan_path))
    processor._init_encoder()
    language = output / 'language'
    language.mkdir()
    records = []
    for instruction in sorted({e['instruction'] for e in manifest['episodes']}):
        text = robotwin_prompt(instruction)
        key = hashlib.sha256(text.encode()).hexdigest()
        embedding = processor._encoder([text], torch.device('cuda:0'))[0].cpu()
        torch.save(embedding, language / f'{key}.pt')
        records.append(dict(text=text, sha256=key, shape=list(embedding.shape)))
    manifest.pop('identity', None)
    manifest['cache_reuse'] = dict(source_manifest=str(source.resolve()), source_sha256=digest(source),
        previous_converter_sha256=manifest['converter_sha256'], previous_spline_sha256=manifest['spline_sha256'],
        all_windows_local_global_max_error=worst)
    manifest['converter_sha256'] = digest(__file__)
    manifest['spline_sha256'] = digest(Path(__file__).with_name('spline.py'))
    manifest['conditioning'] = CONDITIONING
    manifest['language_records'] = records
    manifest['identity'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), conditioning=CONDITIONING, windows_max_error=worst)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--language-from-manifest", type=Path)
    parser.add_argument("--wan-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=["place_can_basket"])
    parser.add_argument('--episodes-per-task',type=int,default=50)
    args = parser.parse_args()
    if args.language_from_manifest:
        if args.wan_path is None:
            parser.error('--wan-path required for language encoding')
        prepare_language(args.language_from_manifest, args.output, args.wan_path)
    else:
        if args.raw_root is None:
            parser.error('--raw-root required for conversion')
        prepare(args.raw_root, args.output, args.tasks,args.episodes_per_task)
