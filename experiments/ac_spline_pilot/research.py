"""Matched adapter tuning and portable research metadata for the existing trainer."""
import dataclasses
import hashlib
import json
from pathlib import Path
from fnmatch import fnmatchcase

import torch
from torch import nn

SCHEMA = "motus-adaptive-research-v1"
DEFAULT_TARGET_MODULES = ("video_model.wan_model.*", "action_expert.*")
PROJECTIONS = ("input_encoder.action_encoder.", "decoder.action_head.")


class LoRALinear(nn.Module):
    def __init__(self, base, rank=64, alpha=128., dropout=0.):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.multiplier = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=5**.5)

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        base = self.base(x.to(self.base.weight.dtype))
        update = torch.nn.functional.linear(torch.nn.functional.linear(self.dropout(x.float()), self.lora_A), self.lora_B)
        return base + (update * self.multiplier).to(base.dtype)


def configure_adapters(model, settings, seed=0):
    """Adapt selected branches; retain the matched full action I/O tuning."""
    settings = dict(settings)
    target_modules = settings.pop('target_modules', DEFAULT_TARGET_MODULES)
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    if not target_modules:
        raise ValueError('LoRA target_modules must contain at least one pattern')
    model.requires_grad_(False)
    # Name-seeded projection reset makes hidden projection initialization match
    # across A/C despite the different boundary widths consuming different RNG.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.startswith('action_expert.') and any(p in name + '.' for p in PROJECTIONS):
            value = int.from_bytes(hashlib.sha256(f'{seed}:{name}'.encode()).digest()[:4], 'little')
            generator = torch.Generator(device=module.weight.device).manual_seed(value)
            nn.init.xavier_uniform_(module.weight, generator=generator)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if 'decoder.action_head' in name:
                nn.init.zeros_(module.weight)
    # Snapshot before replacement so newly added adapter linears are not revisited.
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear)
               and any(fnmatchcase(name, pattern) for pattern in target_modules)]
    if not targets:
        raise ValueError("No linear layers match LoRA target_modules")
    for name, module in targets:
        parent_name, _, child = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, LoRALinear(module, **settings))
    model.action_expert.input_encoder.action_encoder.requires_grad_(True)
    model.action_expert.decoder.action_head.requires_grad_(True)
    # Keep trainable projections in FP32 even when frozen backbone uses BF16.
    model.action_expert.input_encoder.action_encoder.float()
    model.action_expert.decoder.action_head.float()
    model.adapter_targets = [name for name, _ in targets]
    return trainable_summary(model)


def trainable_summary(model, optimizer=None):
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if optimizer is not None:
        optimized = [p for group in optimizer.param_groups for p in group["params"]]
        if len({id(p) for p in optimized}) != len(optimized) or {id(p) for p in optimized} != {id(p) for p in named.values()}:
            raise ValueError("Optimizer must contain every trainable parameter exactly once, and no frozen parameter")
    return dict(trainable_parameters=sum(p.numel() for p in named.values()),
                trainable_tensors=len(named), frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                names={n: list(p.shape) for n, p in named.items()}, adapter_targets=getattr(model, "adapter_targets", []))


def gradient_summary(model):
    report = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            report[name] = None if p.grad is None else dict(finite=bool(torch.isfinite(p.grad).all()), norm=float(p.grad.float().norm()))
    if any(v is not None and not v["finite"] for v in report.values()):
        raise ValueError("Nonfinite trainable gradients")
    return report


def load_compatible_base(model, path):
    """Load before adapter insertion; reset the same complete action I/O in A/C."""
    path = Path(path)
    if path.is_dir():
        candidates = [path / "pytorch_model_0.bin", path / "mp_rank_00_model_states.pt", path / "pytorch_model/mp_rank_00_model_states.pt"]
        path = next((p for p in candidates if p.exists()), candidates[0])
    state = torch.load(path, map_location="cpu", weights_only=False)
    state = state.get("module", state)
    target = model.state_dict()
    def reset(name):
        return any(s in name for s in PROJECTIONS) or name.endswith("pos_embedding")
    retained = {n: t for n, t in state.items() if n in target and not reset(n)}
    bad = [n for n, t in retained.items() if t.shape != target[n].shape]
    missing = [n for n in target if not reset(n) and n not in retained]
    extra = [n for n in state if n not in target and not reset(n)]
    if bad or missing or extra:
        raise ValueError(f"Incompatible base checkpoint: shapes={bad[:8]}, missing={missing[:8]}, unexpected={extra[:8]}")
    model.load_state_dict(retained, strict=False)
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                reset=[n for n in target if reset(n)], loaded_tensors=len(retained))


def load_contract(config):
    """Embed normalization contents rather than relying on mutable inference paths."""
    research = config["research"]
    representation = research["representation"]
    if representation not in ("dense", "spline"):
        raise ValueError("Representation must be dense or spline")
    normalization = json.loads(Path(research["normalization_path"]).read_text())
    scaling = json.loads(Path(research["knot_scaling_path"]).read_text())
    from robotwin_spline import identity, CHANNELS
    from data.robotwin2.adaptive_spline import validate_time_scaling
    if normalization["scope"] != "training-only" or normalization["identity"] != identity({k: v for k, v in normalization.items() if k != "identity"}):
        raise ValueError("Research training requires verified training-only action normalization")
    validate_time_scaling(scaling)
    if scaling["scope"] != "training-only" or scaling["normalization_identity"] != normalization["identity"]:
        raise ValueError("Research training requires matching training-only knot scaling")
    windows = json.loads(Path(research['window_manifest']).read_text())
    from .data import validate_dataset_split
    dataset_split = validate_dataset_split(config, normalization, scaling, windows)
    contract = dict(schema=SCHEMA, representation=representation, dataset_split=dataset_split, normalization=normalization, knot_scaling=scaling,
                    channels=CHANNELS, channel_units=['radian']*6 + ['continuous_open_fraction'] + ['radian']*6 + ['continuous_open_fraction'],
                    degree=3, knot_encoding='observation-relative absolute position', time_units='source_action_step',
                    executed_prefix=dict(source_stride=3, command_count=16, seconds_mapping='requires verified controller clock'),
                    packed_rows=36 if representation == "spline" else config["common"]["num_video_frames"] * config["common"]["video_action_freq_ratio"],
                    target_dim=15 if representation == "spline" else 14, state_dim=14,
                    video_alignment=dict(knot_offset=1, knot_stride=4, future_frames=8),
                    decoder="data.robotwin2.adaptive_spline.decode_local", flow="x_sigma=(1-sigma)*target+sigma*noise; v=noise-target",
                    gripper="continuous fraction; no threshold", lora=research["lora"],
                    window_manifest_sha256=hashlib.sha256(Path(research["window_manifest"]).read_bytes()).hexdigest())
    contract["identity"] = identity(contract)
    return contract


def save_research_bundle(model, directory, resolved_config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = dict(schema=SCHEMA, config=resolved_config, model_config=dataclasses.asdict(model.config),
                    contract=model.research_contract, base=model.research_base, trainables=trainable_summary(model))
    torch.save({n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}, directory / "research_adapter.pt")
    (directory / "research_config.json").write_text(json.dumps(metadata, indent=2))


def load_research_bundle(model, directory):
    directory = Path(directory)
    metadata = json.loads((directory / "research_config.json").read_text())
    if metadata["contract"] != model.research_contract or metadata["base"] != model.research_base:
        raise ValueError("Checkpoint representation/normalization/base identity mismatch")
    params = dict(model.named_parameters())
    state = torch.load(directory / "research_adapter.pt", map_location="cpu", weights_only=True)
    expected = {n for n, p in params.items() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("Checkpoint adapter/projection membership mismatch")
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != params[name].shape:
                raise ValueError(f"Checkpoint projection shape mismatch: {name}")
            params[name].copy_(value)
    return metadata
