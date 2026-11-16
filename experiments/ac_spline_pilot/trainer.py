"""A/C tiny and full-pilot policy layered over the general Motus workflow."""
import json
import logging
import os
from pathlib import Path

import torch
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from train.train import (TrainingFactory, UniDiffuserTrainer, create_model,
                         create_model_and_optimizer, load_config, main)
from .research import (configure_adapters, gradient_summary, load_compatible_base,
                       load_contract, load_research_bundle, save_research_bundle,
                       trainable_summary)


logger = logging.getLogger(__name__)


class ACSplineTrainer(UniDiffuserTrainer):
    def on_gradients_clipped(self):
        if self.global_step >= 3:
            return
        model = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
        gradients = gradient_summary(model)
        if self.rank == 0:
            (self.checkpoint_dir / f'gradients_step_{self.global_step}.json').write_text(
                json.dumps(gradients, indent=2))
        missing = [name for name, value in gradients.items() if value is None]
        if self.global_step == 2:
            if not any(value is not None for value in gradients.values()):
                raise RuntimeError('No trainable gradient path to actual MOTUS losses')
            if missing and self.rank == 0:
                logger.info('%d trainables are unused by this batch; see the gradient report', len(missing))


def create_ac_model(config):
    model = create_model(config)
    model.research_base = load_compatible_base(model, config.finetune.checkpoint_path)
    model.research_contract = OmegaConf.to_container(config.research.contract, resolve=True)
    set_seed(int(config.research.seed))
    configure_adapters(model, OmegaConf.to_container(config.research.lora, resolve=True),
                       seed=int(config.research.seed))
    return model


class ACSplineFactory(TrainingFactory):
    trainer_class = ACSplineTrainer

    def configure_parser(self, parser):
        parser.add_argument('--diagnostic-eval-only', type=str,
                            help='Fresh-process generation from an A/C checkpoint directory')

    def load_config(self, path, args):
        config = load_config(path)
        if not config.get('research', {}).get('enabled', False):
            raise ValueError('The A/C entry point requires an enabled research configuration')
        if (config.common.num_video_frames, config.common.video_action_freq_ratio,
                config.common.global_downsample_rate) != (8, 2, 3):
            raise ValueError('Matched research audits exactly 8 video frames, ratio 2, stride 3')
        representation = config.research.representation
        config.common.action_dim = 15 if representation == 'spline' else 14
        config.common.action_chunk_size = 36 if representation == 'spline' else 16
        config.common.action_representation = representation
        config.common.action_precision = 'float32'
        if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME') or not torch.cuda.is_available():
            raise RuntimeError('Research model execution requires allocated Slurm compute; never use a login node')
        phase_limits = {'tiny': 100, 'pilot': 1000}
        phase = config.research.phase
        if phase not in phase_limits or not 1 <= config.training.max_steps <= phase_limits[phase]:
            raise ValueError('A/C supports tiny runs up to 100 updates or pilot runs up to 1000 updates')
        if config.training_mode != 'finetune':
            raise ValueError('Matched A/C require current observation and state conditioning')
        if args.deepspeed:
            raise ValueError('A/C uses the single-process Accelerator path; DeepSpeed reload is not yet validated')
        for artifact in [config.model.wan.config_path, config.model.wan.vae_path,
                         config.model.vlm.checkpoint_path, config.finetune.checkpoint_path]:
            if not Path(artifact).exists():
                raise FileNotFoundError(f'Research requires existing local model assets: {artifact}')
        config.research.contract = load_contract(OmegaConf.to_container(config, resolve=True))
        from data.robotwin2.research_windows import load_windows
        contract = OmegaConf.to_container(config.research.contract, resolve=True)
        for validation in (False, True):
            windows = load_windows(config.research.window_manifest, validation, contract)
            for window in windows:
                if (window.get('task_name') != config.dataset.task_name
                        and config.dataset.task_name not in Path(window['qpos_path']).parts):
                    raise ValueError(f"Fixed window does not belong to {config.dataset.task_name}: {window['qpos_path']}")
        set_seed(int(config.research.seed))
        # Route fixed windows through the dataset's existing generic params API.
        config.dataset.params = OmegaConf.merge(config.dataset.get('params', {}), {
            'window_manifest': config.research.window_manifest,
            'research_contract': contract,
        })
        return config

    def configure_run(self, config, accelerator):
        if accelerator.num_processes != 1:
            raise ValueError('A/C training is limited to one allocated GPU process')
        if Path(config.system.checkpoint_dir).exists() and not config.resume.checkpoint_path:
            raise FileExistsError('Research output exists; use a fresh --checkpoint_dir to preserve previous diagnostics')

    def create_model_and_optimizer(self, config):
        model, optimizer, scheduler = create_model_and_optimizer(config, model_factory=create_ac_model)
        model.research_trainables = trainable_summary(model, optimizer)
        directory = Path(config.system.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'trainables.json').write_text(json.dumps(model.research_trainables, indent=2))
        return model, optimizer, scheduler

    def register_checkpoint_hooks(self, accelerator, config):
        def save_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                save_research_bundle(accelerator.unwrap_model(models[0]), output_dir,
                                     OmegaConf.to_container(config, resolve=True))
            weights.clear()

        def load_hook(models, input_dir):
            for model in list(models):
                load_research_bundle(accelerator.unwrap_model(model), input_dir)
            models.clear()

        accelerator.register_save_state_pre_hook(save_hook)
        accelerator.register_load_state_pre_hook(load_hook)

    def run(self, trainer, args, config):
        accelerator = trainer.accelerator
        if config.training.scheduler_type == 'linear':
            accelerator.register_for_checkpointing(trainer.scheduler)
        model = accelerator.unwrap_model(trainer.model)
        if args.diagnostic_eval_only:
            load_research_bundle(model, args.diagnostic_eval_only)
        else:
            super().run(trainer, args, config)
        accelerator.wait_for_everyone()
        if trainer.rank == 0:
            from .diagnostic import generate_diagnostic
            generate_diagnostic(model, trainer.train_dataloader, config,
                                Path(config.system.checkpoint_dir) /
                                ('fresh_reload' if args.diagnostic_eval_only else 'generation'),
                                reference_checkpoint=args.diagnostic_eval_only)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main(ACSplineFactory())
