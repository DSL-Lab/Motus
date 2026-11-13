"""LoRA for the concrete MOTUS attention layout, including its fused QKV tensor.

No PEFT support is installed in the verified environment. Keep base parameters
frozen and expose a single matching adapter recipe for A and C.
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize


class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, alpha=32):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

    def merged(self):
        result = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None,
                           device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            result.weight.copy_(self.base.weight + (self.b @ self.a) * self.scale)
            if result.bias is not None:
                result.bias.copy_(self.base.bias)
        return result


class FusedQKVLoRA(nn.Module):
    """Independent rank-r updates for Q/K/V in [3,heads,in,head_dim]."""
    def __init__(self, weight, rank=16, alpha=32):
        super().__init__()
        if weight.ndim != 4 or weight.shape[0] != 3:
            raise ValueError("expected MOTUS fused QKV [3,heads,in,head_dim]")
        self.heads, self.input_dim, self.head_dim = weight.shape[1:]
        self.scale = alpha / rank
        self.a = nn.Parameter(weight.new_empty(3, rank, self.input_dim))
        self.b = nn.Parameter(weight.new_zeros(3, self.heads * self.head_dim, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, weight):
        update = (self.b @ self.a).reshape(3, self.heads, self.head_dim, self.input_dim)
        return weight + update.permute(0, 1, 3, 2) * self.scale


def configure_lora(model, rank=16, alpha=32):
    """Freeze backbone, target actual attention projections, train action I/O."""
    if isinstance(rank,bool) or not isinstance(rank,int) or rank<1 or alpha<=0:
        raise ValueError('LoRA rank must be a positive integer and alpha positive')
    model.requires_grad_(False)
    targets = []
    for index, block in enumerate(model.video_model.wan_model.blocks):
        for attention in ("self_attn", "cross_attn"):
            module = getattr(block, attention)
            for name in ("q", "k", "v", "o"):
                layer = getattr(module, name)
                if not isinstance(layer, nn.Linear):
                    raise TypeError(f"unexpected WAN attention: {attention}.{name}")
                setattr(module, name, LoRALinear(layer,rank=rank,alpha=alpha))
                targets.append(f"video_model.wan_model.blocks.{index}.{attention}.{name}")
    for index, block in enumerate(model.action_expert.blocks):
        parametrize.register_parametrization(block, "wan_action_qkv", FusedQKVLoRA(block.wan_action_qkv,rank=rank,alpha=alpha))
        block.wan_action_o = LoRALinear(block.wan_action_o,rank=rank,alpha=alpha)
        targets.extend([f"action_expert.blocks.{index}.wan_action_qkv",
                        f"action_expert.blocks.{index}.wan_action_o"])
    model.action_expert.input_encoder.action_encoder.requires_grad_(True)
    model.action_expert.decoder.action_head.requires_grad_(True)
    counts = defaultdict(lambda: {"trainable": 0, "frozen": 0})
    for name, p in model.named_parameters():
        component = name.split(".")[0]
        counts[component]["trainable" if p.requires_grad else "frozen"] += p.numel()
    return {"rank": rank, "alpha": alpha, "scale": alpha/rank, "dropout": 0.0, "targets": targets, "counts": dict(counts),
            "fully_trained": ["action_expert.input_encoder.action_encoder", "action_expert.decoder.action_head"]}


def parameter_groups(model, lr=1e-4, weight_decay=0.01):
    groups = {True: [], False: []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups[p.ndim >= 2 and not name.endswith("bias") and "norm" not in name].append(p)
    return [{"params": values, "lr": lr, "weight_decay": weight_decay if decay else 0.0}
            for decay, values in groups.items() if values]


def merge_lora(model):
    """Same inference merge policy for A/C. Merge only once, before evaluation."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
        if parametrize.is_parametrized(parent, "wan_action_qkv"):
            parametrize.remove_parametrizations(parent, "wan_action_qkv", leave_parametrized=True)


def load_initial_weights(model, path, report_path):
    """Load every compatible tensor, explicitly adapt C's two changed projections.

    Reject unexpected omissions, including missing state projections. Only the
    deterministic C positional buffer may be regenerated without checkpoint data.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "mp_rank_00_model_states.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    source = checkpoint.get("module", checkpoint)
    target = model.state_dict()
    loaded, adapted, mismatched, unexpected = {}, {}, {}, []
    spline = model.config.action_representation == "spline"
    for name, value in source.items():
        # ActionModule registers the same expert a second time. Adapt both state
        # dict aliases identically rather than silently dropping either path.
        canonical = name.removeprefix('action_module.') if name.startswith('action_module.action_expert.') else name
        if name not in target:
            unexpected.append(name)
        elif value.shape == target[name].shape:
            loaded[name] = value
        elif spline and canonical == "action_expert.input_encoder.action_encoder.0.weight" and value.shape[1] == 14:
            tensor = torch.zeros_like(target[name], device="cpu")
            tensor[:, 1:] = value
            loaded[name] = tensor
            adapted[name] = "prepend zero knot column, copy 14 action columns"
        elif spline and canonical.startswith("action_expert.decoder.action_head.") and value.shape[0] == 14 and target[name].shape[0] == 15:
            tensor = torch.zeros_like(target[name], device="cpu")
            tensor[1:] = value
            loaded[name] = tensor
            adapted[name] = "prepend zero knot output, copy 14 action outputs"
        elif spline and canonical == "action_expert.input_encoder.pos_embedding":
            adapted[name] = "regenerate deterministic positional buffer for 36 rows"
        else:
            mismatched[name] = {"checkpoint": list(value.shape), "model": list(target[name].shape)}
    missing = sorted(set(target) - set(loaded))
    report = {"source": str(path), "loaded": sorted(loaded), "adapted": adapted,
              "missing": missing, "unexpected": unexpected, "shape_mismatched": mismatched}
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    allowed_missing = {"action_expert.input_encoder.pos_embedding", "action_module.action_expert.input_encoder.pos_embedding"} if spline else set()
    if unexpected or mismatched or set(missing) - allowed_missing:
        raise ValueError(f"initial checkpoint incompatible; see {report_path}")
    model.load_state_dict(loaded, strict=False)
    return report
