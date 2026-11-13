"""Full pretrained A/C smoke on allocated hardware, not policy-quality evidence."""
import gc
import hashlib
import json
import os
import random
import numpy as np
from pathlib import Path
import sys
import time

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'train'))
from data.dataset import create_dataset, collate_fn
from train import create_model_and_optimizer


def language_cache(config):
    from data.robotwin2.robotwin_data_convert.robotwin_converter import T5EmbeddingProcessor
    from utils.robotwin_prompt import robotwin_prompt
    manifest_path = Path(config.dataset.episode_manifest)
    manifest = json.loads(manifest_path.read_text())
    output = manifest_path.parent / 'language'
    output.mkdir(exist_ok=True)
    processor = T5EmbeddingProcessor(str(ROOT / '.runtime/model-metadata/Wan2.2-TI2V-5B'))
    for instruction in sorted({e['instruction'] for e in manifest['episodes']}):
        instruction = robotwin_prompt(instruction)
        path = output / (hashlib.sha256(instruction.encode()).hexdigest() + '.pt')
        if path.exists():
            continue
        processor._init_encoder()
        with torch.no_grad():
            embedding = processor._encoder([instruction], torch.device('cuda:0'))[0].cpu()
        torch.save(embedding, path)
    del processor
    gc.collect()
    torch.cuda.empty_cache()


def main(representation=None):
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('requires allocated GPU')
    config = OmegaConf.load(ROOT / 'configs/robotwin_ac.yaml')
    config.training.lora_rank = 16  # Interface repair phase, not the subsequent rank-256 experiment.
    output = ROOT / 'outputs' / f'model-smoke-{os.environ["SLURM_JOB_ID"]}'
    output.mkdir(exist_ok=False)
    language_cache(config)
    results = []
    representations = ('dense', 'spline') if representation is None else (('dense' if representation == 'A' else 'spline'),)
    for representation in representations:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        config.common.action_representation = representation
        config.system.checkpoint_dir = str(output / representation)
        batch = next(iter(DataLoader(create_dataset(config), batch_size=1, collate_fn=collate_fn)))
        model, optimizer, scheduler = create_model_and_optimizer(config)
        model.train()
        arguments = dict(first_frame=batch['first_frame'].cuda().bfloat16(),
                         video_frames=batch['video_frames'].cuda().bfloat16(),
                         state=batch['initial_state'].cuda().bfloat16(),
                         actions=batch['action_sequence'].cuda().bfloat16(),
                         language_embeddings=batch['language_embedding'].cuda().bfloat16(),
                         vlm_inputs={k: v.cuda() if torch.is_tensor(v) else v for k,v in batch['vlm_inputs'].items()})
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        losses = model.training_step(**arguments)
        loss = losses['total_loss']
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        missing = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is None]
        invalid = [n for n,p in model.named_parameters() if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
        if missing or invalid:
            raise ValueError(f'gradient audit: missing={missing}, invalid={invalid}')
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        torch.cuda.synchronize()
        result = {'representation': representation, 'loss': float(loss),
                  'step_s': time.perf_counter() - start,
                  'peak_memory_bytes': torch.cuda.max_memory_allocated(),
                  'batch_shape': list(batch['action_sequence'].shape)}
        print(json.dumps(result), flush=True)
        results.append(result)
        (output / 'results.json').write_text(json.dumps(results, indent=2))
        del model, optimizer, scheduler, batch, arguments, losses, loss
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("representation", choices=["A", "C"], nargs="?")
    main(parser.parse_args().representation)
