"""Small LoRA adapters for Motus's packed action QKV tensor and attention O."""

import math
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, value):
        return self.base(value) + (value @ self.lora_a.T) @ self.lora_b.T


class ActionQKVLoRA(nn.Module):
    def __init__(self, packed_weight, rank):
        super().__init__()
        _, self.heads, width, self.head_dim = packed_weight.shape
        self.lora_a = nn.Parameter(packed_weight.new_empty(3, width, rank))
        self.lora_b = nn.Parameter(packed_weight.new_zeros(3, rank, self.heads * self.head_dim))
        nn.init.normal_(self.lora_a, std=1 / math.sqrt(width))

    def forward(self, value):
        update = torch.einsum("btd,kdr,kre->kbte", value, self.lora_a, self.lora_b)
        return update.unflatten(-1, (self.heads, self.head_dim))


def configure_lora(model, rank=8, video_lora=False):
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    model.requires_grad_(False)
    for block in model.action_expert.blocks:
        block.wan_action_lora = ActionQKVLoRA(block.wan_action_qkv, rank)
        block.wan_action_o = LoRALinear(block.wan_action_o, rank)
    if video_lora:
        for block in model.video_model.wan_model.blocks:
            for name in ("q", "k", "v", "o"):
                setattr(block.self_attn, name, LoRALinear(getattr(block.self_attn, name), rank))
    # Only the resized input and output projections are newly required components.
    model.action_expert.input_encoder.action_encoder[0].requires_grad_(True)
    model.action_expert.decoder.action_head[-1].requires_grad_(True)
    return [name for name, value in model.named_parameters() if value.requires_grad]


def check_gradients(model):
    """Require finite gradients on every adapter/new projection and frozen bases.

    Zero gradients on A at initialization are expected because B starts at zero.
    At least one nonzero gradient per trainable component is still required.
    """
    groups = {}
    for name, value in model.named_parameters():
        if not value.requires_grad:
            if value.grad is not None:
                raise RuntimeError(f"Frozen weight received a gradient: {name}")
            continue
        if value.grad is None or not torch.isfinite(value.grad).all():
            raise RuntimeError(f"Missing or nonfinite gradient: {name}")
        group = name.rsplit(".", 1)[0]
        groups[group] = groups.get(group, 0) + value.grad.float().square().sum().item()
    if any(norm == 0 for norm in groups.values()):
        raise RuntimeError(f"Trainable component has zero gradient: {groups}")
    return {name: math.sqrt(norm) for name, norm in groups.items()}
