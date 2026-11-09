"""Initialize direct spline prediction from a dense Motus checkpoint."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .lora import configure_lora


def adapt_action_representation(model, metadata):
    from models.action_expert import get_1d_sincos_pos_embed_from_grid

    if metadata["num_video_frames"] != model.config.num_video_frames:
        raise ValueError("Cache must preserve the dense checkpoint's RGB frame count")
    expert = model.action_expert
    action_dim = len(metadata["action_min"])
    if expert.config.action_dim != action_dim or model.config.training_mode != "finetune":
        raise ValueError("Use a dense Motus finetune checkpoint for the same robot action space")
    old_input = expert.input_encoder.action_encoder[0]
    old_output = expert.decoder.action_head[-1]
    new_input = nn.Linear(action_dim + 1, old_input.out_features).to(old_input.weight)
    new_output = nn.Linear(old_output.in_features, action_dim + 1).to(old_output.weight)
    minimum = old_input.weight.new_tensor(metadata["action_min"])
    maximum = old_input.weight.new_tensor(metadata["action_max"])
    half_span, midpoint = (maximum - minimum) / 2, (maximum + minimum) / 2
    with torch.no_grad():
        # Dense RoboTwin training uses physical qpos. Transfer the old projection
        # through the new [-1,1] control-point affine normalization.
        new_input.weight[:, 0].zero_()
        new_input.weight[:, 1:].copy_(old_input.weight * half_span)
        new_input.bias.copy_(old_input.bias + old_input.weight @ midpoint)
        inverse_span = torch.where(half_span > 0, 1 / half_span.clamp_min(1e-8), 0)
        new_output.weight[1:].copy_(old_output.weight * inverse_span[:, None])
        new_output.bias[1:].copy_(old_output.bias * inverse_span)
        nn.init.normal_(new_output.weight[:1], std=0.01)
        new_output.bias[:1].zero_()
    expert.input_encoder.action_encoder[0] = new_input
    expert.decoder.action_head[-1] = new_output
    rows = int(metadata["rows"])
    expert.config.action_dim = action_dim + 1
    expert.config.chunk_size = rows + 1
    # Ordinal parameter-row positions remain available throughout denoising.
    # Ground-truth physical knot times are targets, never side inputs.
    positions = get_1d_sincos_pos_embed_from_grid(
        expert.config.dim, np.arange(rows + 1 + expert.config.num_registers))
    expert.input_encoder.pos_embedding = positions[None].to(old_input.weight)
    model.config.action_dim = action_dim + 1
    model.config.action_chunk_size = rows
    model.spline_metadata = metadata


def build_spline_model(checkpoint, wan_path, vlm_path, metadata, *, rank=8, video_lora=False, device="cuda"):
    from models.motus import Motus, MotusConfig

    path = Path(checkpoint)
    directory = path if path.is_dir() else path.parent
    saved = json.loads((directory / "config.json").read_text())
    common = saved["common"]
    action = saved.get("action_expert", saved.get("model", {}).get("action_expert"))
    und = saved.get("und_expert", saved.get("model", {}).get("und_expert"))
    config = MotusConfig(
        device=device, load_pretrained_backbones=False,
        wan_checkpoint_path=str(wan_path), wan_config_path=str(wan_path),
        vae_path=str(Path(wan_path) / "Wan2.2_VAE.pth"), vlm_checkpoint_path=str(vlm_path),
        action_dim=common["action_dim"], action_state_dim=common["state_dim"],
        action_expert_dim=action["hidden_size"],
        action_expert_ffn_dim_multiplier=action["ffn_dim_multiplier"], action_expert_norm_eps=action["norm_eps"],
        und_expert_hidden_size=und["hidden_size"], und_expert_ffn_dim_multiplier=und["ffn_dim_multiplier"],
        und_expert_norm_eps=und["norm_eps"], vlm_adapter_input_dim=und["vlm"]["input_dim"],
        vlm_adapter_projector_type=und["vlm"]["projector_type"],
        num_video_frames=common["num_video_frames"], video_height=common["video_height"],
        video_width=common["video_width"], global_downsample_rate=common["global_downsample_rate"],
        video_action_freq_ratio=common["video_action_freq_ratio"], batch_size=1)
    model = Motus(config)
    model.load_checkpoint(str(checkpoint), strict=True)
    adapt_action_representation(model, metadata)
    names = configure_lora(model, rank, video_lora)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} in {len(names)} tensors")
    return model


def save_adapter(path, model, *, rank, video_lora, step):
    trainable = {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}
    torch.save(dict(version=1, parameters=trainable, metadata=model.spline_metadata,
                    rank=rank, video_lora=video_lora, step=step), path)


def load_adapter(model, saved):
    parameters = dict(model.named_parameters())
    expected = {name for name, value in parameters.items() if value.requires_grad}
    if expected != set(saved["parameters"]):
        raise ValueError("Adapter keys do not match the configured trainable parameters")
    if model.spline_metadata != saved["metadata"]:
        raise ValueError("Adapter and dataset normalization/timeline metadata differ")
    with torch.no_grad():
        for name, value in saved["parameters"].items():
            parameters[name].copy_(value.to(parameters[name]))
