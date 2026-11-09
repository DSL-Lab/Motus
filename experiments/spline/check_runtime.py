"""Check dependencies on CPU; check full deployment imports in a GPU allocation."""

import argparse
import importlib
import importlib.metadata
import importlib.util
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    policy = root / "inference/robotwin/Motus"
    sys.path[:0] = [str(policy), str(root), str(root / "bak")]
    import torch

    _ = importlib.import_module("scipy.interpolate").make_splprep
    transformers = importlib.import_module("transformers")
    _ = transformers.AutoProcessor
    _ = transformers.Qwen3VLForConditionalGeneration
    for name in (
        "accelerate",
        "cv2",
        "deepspeed.comm.comm",
        "diffusers",
        "einops",
        "ftfy",
        "flash_attn",
        "matplotlib",
        "omegaconf",
        "safetensors",
        "sentencepiece",
    ):
        importlib.import_module(name)
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "transformers",
        "diffusers",
        "deepspeed",
        "flash-attn",
    ):
        print(f"{name}=={importlib.metadata.version(name)}")
    if args.require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside Apptainer")
        for name in ("sapien", "mplib", "curobo"):
            importlib.import_module(name)
        # The existing Wan T5 class initializes CUDA while importing. Preserve
        # that inference code, and check it only in the allocated GPU job.
        spec = importlib.util.spec_from_file_location(
            "_motus_import_check", policy / "deploy_policy.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        attention = importlib.import_module("wan.modules.attention")
        if not (attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE):
            raise RuntimeError(
                "The deployed Motus attention code requires FlashAttention"
            )
        # Exercise the new binary before paying the model's loading cost.
        flash = importlib.import_module("flash_attn")
        query = torch.zeros((1, 2, 1, 64), device="cuda", dtype=torch.bfloat16)
        result = flash.flash_attn_func(query, query, query)
        torch.cuda.synchronize()
        if result.shape != query.shape or not torch.isfinite(result).all():
            raise RuntimeError("FlashAttention GPU smoke check failed")
        print("GPU:", torch.cuda.get_device_name(0))
        print("Full Motus deployment imports OK.")
    else:
        print("CPU dependencies OK; full deployment import requires the GPU job.")
    print("No model or weights were loaded.")


if __name__ == "__main__":
    main()
