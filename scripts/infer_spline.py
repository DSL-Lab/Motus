"""Predict spline parameters and decode controller commands for one cached observation."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from transformers import AutoProcessor

from data.dataset import collate_fn
from motus_actions.spline_cache import SplineDataset
from motus_actions.spline_model import build_spline_model, load_adapter
from motus_actions.spline_window import decode_window
from scripts.train_spline import move_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wan-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--execution-rows", type=int, default=12)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode-rgb", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    saved = torch.load(args.adapter, map_location="cpu", weights_only=True)
    model = build_spline_model(args.checkpoint, args.wan_path, args.vlm_path, saved["metadata"],
                               rank=saved["rank"], video_lora=saved["video_lora"], device=args.device)
    load_adapter(model, saved)
    model.eval()
    dataset = SplineDataset(args.cache, AutoProcessor.from_pretrained(args.vlm_path),
                            (model.config.video_height, model.config.video_width))
    if dataset.metadata != saved["metadata"]:
        raise ValueError("Inference cache differs from adapter metadata")
    batch = move_batch(collate_fn([dataset[args.sample]]), args.device)
    torch.manual_seed(args.seed)
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode(), torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16):
        frames, parameters = model.inference_step(
            first_frame=batch["first_frame"], state=batch["initial_state"],
            language_embeddings=batch["language_embedding"], vlm_inputs=batch["vlm_inputs"],
            num_inference_steps=args.denoise_steps, decode_rgb=args.decode_rgb)
    if torch.device(args.device).type == "cuda":
        torch.cuda.synchronize()
    commands, ticks, next_call = decode_window(parameters[0].float().cpu().numpy(),
                                              saved["metadata"], args.execution_rows)
    latency = time.perf_counter() - start
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = dict(parameters=parameters.float().cpu().numpy(), commands=commands,
                  command_times=ticks, next_call_seconds=next_call, inference_seconds=latency)
    if frames is not None:
        values["rgb"] = frames.float().cpu().numpy()
    np.savez_compressed(output, **values)
    print(f"{len(commands)} commands at {saved['metadata']['controller_hz']} Hz; next call after {next_call:.6g}s; latency {latency:.3f}s")


if __name__ == "__main__":
    main()
