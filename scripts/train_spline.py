"""Bounded LoRA training / tiny overfit for direct Motus spline parameters."""

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from data.dataset import collate_fn
from motus_actions.lora import check_gradients
from motus_actions.spline_cache import SplineDataset
from motus_actions.spline_model import build_spline_model, load_adapter, save_adapter


def move_batch(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {name: move_batch(item, device) for name, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wan-path", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--tiny-samples", type=int, help="Repeat only the first N cached windows")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--video-lora", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", help="Adapter only; optimizer is restarted")
    parser.add_argument("--fixed-noise", action="store_true", help="Debug tiny overfit with identical FM noise/times")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or (args.tiny_samples is not None and args.tiny_samples < 1):
        parser.error("steps, batch size, and tiny sample count must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    metadata = json.loads((Path(args.cache) / "metadata.json").read_text())
    model = build_spline_model(args.checkpoint, args.wan_path, args.vlm_path, metadata,
                               rank=args.rank, video_lora=args.video_lora, device=args.device)
    if args.resume:
        load_adapter(model, torch.load(args.resume, map_location="cpu", weights_only=True))
    processor = AutoProcessor.from_pretrained(args.vlm_path)
    dataset = SplineDataset(args.cache, processor, (model.config.video_height, model.config.video_width), args.tiny_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=not args.fixed_noise,
                        num_workers=0, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    model.train()
    model.vlm_model.eval()
    model.video_model.vae.model.eval()
    batches = iter(loader)
    history = []
    for step in range(args.steps):
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(loader)
            batch = next(batches)
        batch = move_batch(batch, args.device)
        if args.fixed_noise:
            torch.manual_seed(args.seed)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16):
            losses = model.training_step(
                first_frame=batch["first_frame"], video_frames=batch["video_frames"],
                state=batch["initial_state"], actions=batch["action_sequence"],
                language_embeddings=batch["language_embedding"], vlm_inputs=batch["vlm_inputs"],
                video_frame_valid=batch["video_frame_valid"])
        losses["total_loss"].backward()
        if step == 0:
            norms = check_gradients(model)
            print(f"Verified finite gradients in {len(norms)} trainable components")
            if not losses["video_loss"].requires_grad:
                raise RuntimeError("Joint video loss was detached")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        metrics = {name: float(losses[name].detach()) for name in ("total_loss", "video_loss", "action_loss")}
        history.append(dict(step=step + 1, **metrics))
        print(json.dumps(history[-1]), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_adapter(output, model, rank=args.rank, video_lora=args.video_lora, step=args.steps)
    output.with_suffix(".losses.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
