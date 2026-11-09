"""Cached RoboTwin-format demonstrations for direct spline-parameter training."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .spline_window import fit_adaptive_spline, local_window, normalize_actions, select_video_frames


def load_actions(path, source_fps):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        actions = np.asarray(value["actions"], dtype=float)
        times = np.asarray(value["timestamps"], dtype=float)
    else:
        actions = np.asarray(value, dtype=float)
        times = np.arange(len(actions)) / source_fps
    if (actions.ndim != 2 or times.shape != (len(actions),) or len(actions) < 2
            or not np.isfinite(actions).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError(f"Invalid actions or timestamps in {path}")
    return times, actions


def prepare_cache(args):
    from decord import VideoReader, cpu

    root, output = Path(args.dataset_dir).resolve(), Path(args.cache).resolve()
    if (args.source_fps <= 0 or args.controller_hz <= 0 or args.observation_stride < 1
            or args.num_video_frames < 4 or args.num_video_frames % 4
            or args.rows < 3 + args.num_video_frames * args.knot_stride + 4):
        raise ValueError("Require positive rates/stride, future frames divisible by 4, and enough spline rows")
    episodes = sorted(root.glob(f"{args.split}/{args.task}/qpos/*.pt"))
    if args.max_episodes:
        episodes = episodes[:args.max_episodes]
    if not episodes:
        raise ValueError(f"No qpos episodes under {root}")
    sources = []
    for path in episodes:
        task = path.parent.parent
        files = [path, task / "videos" / f"{path.stem}.mp4",
                 task / "umt5_wan" / path.name, task / "metas" / f"{path.stem}.txt"]
        for file in files:
            stat = file.stat()  # missing paired assets fail preprocessing explicitly
            sources.append((str(file), stat.st_size, stat.st_mtime_ns))
    settings = {key: value for key, value in vars(args).items() if key != "cache"}
    fingerprint = hashlib.sha256(json.dumps([settings, sources], sort_keys=True).encode()).hexdigest()
    manifest = output / "metadata.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Cache settings or sources changed; choose a new cache directory")
        if all((output / item["cache_file"]).exists() for item in previous["episodes"]):
            print(f"Reusing {len(previous['episodes'])} cached episodes: {output}")
            return
    loaded = [load_actions(path, args.source_fps) for path in episodes]
    dimensions = {actions.shape[1] for _, actions in loaded}
    if len(dimensions) != 1 or any(i < 0 or i >= next(iter(dimensions)) for i in args.gripper_indices):
        raise ValueError("Inconsistent action dimensions or invalid gripper indices")
    minimum = np.min(np.stack([actions.min(axis=0) for _, actions in loaded]), axis=0)
    maximum = np.max(np.stack([actions.max(axis=0) for _, actions in loaded]), axis=0)
    output.mkdir(parents=True, exist_ok=True)
    records, time_scale = [], 0.0
    for number, (path, (times, actions)) in enumerate(zip(episodes, loaded)):
        task = path.parent.parent
        video = task / "videos" / f"{path.stem}.mp4"
        reader = VideoReader(str(video), ctx=cpu(0), num_threads=1)
        frame_times = np.asarray(reader.get_frame_timestamp(np.arange(len(reader))))[:, 0].astype(float)
        normalized = normalize_actions(actions, minimum, maximum)
        fitted = fit_adaptive_spline(times, normalized, args.tolerance)
        error = float(np.max(np.abs(fitted(times) - normalized)))
        if error > args.tolerance + 1e-7:
            raise ValueError(f"Spline reconstruction failed for {path}: {error}")
        sample_dt = float(np.median(np.diff(times)))
        observations = np.arange(0, len(times) - 1, args.observation_stride)
        observations = observations[(times[observations] >= frame_times[0]) & (times[observations] <= frame_times[-1])]
        if not len(observations):
            raise ValueError(f"No aligned observations for {path}")
        windows, indices, masks = [], [], []
        for index in observations:
            window = local_window(fitted.t, fitted.c, times[index], args.rows, sample_dt)
            selected, valid = select_video_frames(window, times[index], frame_times,
                                                   args.num_video_frames, args.knot_stride)
            windows.append(window)
            indices.append(selected)
            masks.append(valid)
        windows = np.asarray(windows)
        time_scale = max(time_scale, float(np.abs(windows[:, :, 0]).max()))
        name = f"episode_{number:06d}.npz"
        np.savez_compressed(output / name, parameters=windows.astype(np.float32),
                            state=actions[observations].astype(np.float32),
                            frame_indices=indices, frame_valid=masks,
                            observation_times=times[observations], frame_times=frame_times,
                            knots=fitted.t, controls=fitted.c, action_times=times,
                            reconstruction_max_abs=error)
        records.append({"cache_file": name, "video": str(video), "samples": len(observations),
                        "language": str(task / "umt5_wan" / path.name),
                        "instructions": str(task / "metas" / f"{path.stem}.txt")})
        print(f"{path}: {len(fitted.c)} controls, {len(observations)} windows, max error {error:.6g}")
    metadata = dict(settings, version=1, fingerprint=fingerprint, episodes=records,
                    action_min=minimum.tolist(), action_max=maximum.tolist(),
                    time_scale=max(time_scale, 1 / args.controller_hz))
    metadata["max_duration"] = args.max_duration
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved cache: {manifest}")


class SplineDataset(Dataset):
    """Deterministic indexable windows; optional first N samples for tiny overfit."""

    def __init__(self, cache, processor, video_size, limit=None):
        self.cache = Path(cache)
        self.metadata = json.loads((self.cache / "metadata.json").read_text())
        self.processor, self.video_size = processor, video_size
        self.episodes = [dict(np.load(self.cache / item["cache_file"], allow_pickle=False))
                         for item in self.metadata["episodes"]]
        self.index = [(episode, row) for episode, data in enumerate(self.episodes)
                      for row in range(len(data["parameters"]))]
        if limit is not None:
            self.index = self.index[:limit]
        if not self.index:
            raise ValueError("Empty spline dataset")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        from data.utils.image_utils import load_video_frames, tensor_to_pil
        from utils.vlm_utils import preprocess_vlm_messages

        episode, row = self.index[index]
        data, record = self.episodes[episode], self.metadata["episodes"][episode]
        frames = load_video_frames(record["video"], data["frame_indices"][row].tolist(), self.video_size)
        parameters = data["parameters"][row].copy()
        parameters[:, 0] /= self.metadata["time_scale"]
        language = torch.load(record["language"], map_location="cpu", weights_only=True)
        # Select matching text and T5 entry deterministically for reproducible tiny fits.
        language = language[0] if isinstance(language, list) else language
        instruction = Path(record["instructions"]).read_text().splitlines()[0].strip()
        return {"first_frame": frames[0], "video_frames": frames[1:],
                "initial_state": torch.from_numpy(data["state"][row]),
                "action_sequence": torch.from_numpy(parameters),
                "video_frame_valid": torch.from_numpy(data["frame_valid"][row]),
                "language_embedding": language.squeeze(0) if language.ndim == 3 else language,
                "vlm_inputs": preprocess_vlm_messages(instruction, tensor_to_pil(frames[0]), self.processor)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--split", default="clean", choices=["clean", "randomized"])
    parser.add_argument("--task", default="*")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--source-fps", type=float, default=30)
    parser.add_argument("--controller-hz", type=float, default=30)
    parser.add_argument("--observation-stride", type=int, default=3)
    parser.add_argument("--rows", type=int, default=24)
    parser.add_argument("--num-video-frames", type=int, default=8, help="Future RGB frames, before VAE compression")
    parser.add_argument("--knot-stride", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--gripper-indices", type=int, nargs="*", default=[6, 13])
    parser.add_argument("--max-duration", type=float, default=10, help="Deployment cap in seconds; never retimes motion")
    prepare_cache(parser.parse_args())


if __name__ == "__main__":
    main()
