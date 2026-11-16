"""Check shared factory behavior and A/C policy without loading model backbones."""
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from omegaconf import OmegaConf

from experiments.ac_spline_pilot.tests.test_research import ROOT, definitions


class FactoryTests(unittest.TestCase):
    def setUp(self):
        self.namespace = dict(
            OmegaConf=OmegaConf, Path=Path, os=os, torch=torch,
            logger=logging.getLogger(__name__), UniDiffuserTrainer=object,
            create_model=Mock(),
        )
        definitions('train/train.py', {'load_config', 'TrainingFactory',
                                      'create_model_and_optimizer', 'reporting_backends'}, self.namespace)
        self.namespace.update(ACSplineTrainer=object, load_contract=Mock(), set_seed=Mock())
        definitions('experiments/ac_spline_pilot/trainer.py', {'ACSplineFactory'}, self.namespace)
        self.config_path = ROOT / 'experiments/ac_spline_pilot/configs/robotwin_research_C.yaml'

    def test_shared_factory_does_not_enforce_ac_phase_or_frame_limits(self):
        config = self.namespace['load_config'](str(ROOT / 'configs/robotwin.yaml'))
        config.research = {'enabled': True, 'phase': 'pilot', 'representation': 'spline'}
        config.training.max_steps = 1000
        config.common.num_video_frames = 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'generic.yaml'
            OmegaConf.save(config, path)
            args = SimpleNamespace(deepspeed='ds_config.json')
            result = self.namespace['TrainingFactory']().load_config(str(path), args)
        self.assertEqual(result.training.max_steps, 1000)
        self.assertEqual(result.common.action_chunk_size, 8)
        self.assertEqual(result.common.action_dim, 14)

    def test_ac_factory_loads_moved_inheritance_and_supplies_window_params(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {name: directory for name in (
                'MOTUS_WINDOWS', 'MOTUS_NORMALIZATION', 'MOTUS_KNOT_SCALING',
                'MOTUS_DATASET_ROOT', 'MOTUS_WAN', 'MOTUS_VAE', 'MOTUS_VLM',
                'MOTUS_BASE_CHECKPOINT')}
            env.update(SLURM_JOB_ID='test', SLURMD_NODENAME='test')
            with patch.dict(os.environ, env), patch('torch.cuda.is_available', return_value=True), \
                    patch('data.robotwin2.research_windows.load_windows', return_value=[{
                        'qpos_path': str(Path(directory) / 'place_can_basket/qpos/0.pt')}]) as windows:
                profiles = [('research', 'A', (14, 16), 100), ('research', 'C', (15, 36), 100),
                            ('pilot', 'A', (14, 16), 1000), ('pilot', 'C', (15, 36), 1000)]
                for profile, variant, shape, steps in profiles:
                    self.namespace['load_contract'].return_value = {'representation': 'dense' if variant == 'A' else 'spline'}
                    path = self.config_path.with_name(f'robotwin_{profile}_{variant}.yaml')
                    result = self.namespace['ACSplineFactory']().load_config(
                        str(path), SimpleNamespace(deepspeed=None))
                    self.assertEqual((result.common.action_dim, result.common.action_chunk_size), shape)
                    self.assertEqual(result.common.action_precision, 'float32')
                    self.assertEqual(result.training.max_steps, steps)
                    self.assertEqual(result.training.gradient_accumulation_steps, 16 if profile == 'pilot' else 1)
                    self.assertEqual(result.dataset.params.window_manifest, directory)
                    self.assertEqual(result.dataset.params.research_contract, result.research.contract)
                    self.assertEqual(result.research.lora.rank, 64)
                    self.assertEqual(result.research.lora.alpha, 128.)
                    self.assertEqual(list(result.research.lora.target_modules),
                                     ['video_model.wan_model.*', 'action_expert.*'])
                    self.assertEqual(result.dataset.task_mode, 'single')
                    self.assertEqual(result.dataset.task_name, 'place_can_basket')
                    self.assertEqual(result.dataset.data_mode, 'clean')
                    self.assertEqual(result.dataset.max_episodes, 50)
                    self.assertEqual(set(result.dataset.params), {'window_manifest', 'research_contract'})
                self.assertEqual(windows.call_count, 8)
                oversized = self.namespace['load_config'](str(self.config_path))
                oversized.research.phase = 'pilot'
                oversized.training.max_steps = 1001
                oversized_path = Path(directory) / 'oversized.yaml'
                OmegaConf.save(oversized, oversized_path)
                with self.assertRaisesRegex(ValueError, 'up to 1000'):
                    self.namespace['ACSplineFactory']().load_config(
                        str(oversized_path), SimpleNamespace(deepspeed=None))
                windows.return_value = [{'qpos_path': str(Path(directory) / 'other_task/qpos/0.pt')}]
                with self.assertRaisesRegex(ValueError, 'does not belong to place_can_basket'):
                    self.namespace['ACSplineFactory']().load_config(
                        str(self.config_path), SimpleNamespace(deepspeed=None))
                with self.assertRaisesRegex(ValueError, 'DeepSpeed'):
                    self.namespace['ACSplineFactory']().load_config(
                        str(self.config_path), SimpleNamespace(deepspeed='ds_config.json'))

    def test_none_logging_is_supported_by_accelerator_without_tracker_dependencies(self):
        from accelerate import Accelerator
        backends = self.namespace['reporting_backends']('none')
        self.assertEqual(backends, [])
        accelerator = Accelerator(cpu=True, log_with=backends or None)
        self.assertFalse(accelerator.trackers)
        self.assertEqual(self.namespace['reporting_backends']('all'), ['wandb', 'tensorboard'])
        self.assertEqual(self.namespace['reporting_backends']('tensorboard'), ['tensorboard'])

    def test_optimizer_uses_custom_model_trainables_and_separate_wan_lr(self):
        model = nn.Module()
        model.video_model = nn.Module()
        model.video_model.wan_model = nn.Linear(2, 2)
        model.projection = nn.Linear(2, 1)
        model.frozen = nn.Linear(2, 1).requires_grad_(False)
        factory = Mock(return_value=model)
        scheduler = object()
        self.namespace['create_scheduler'] = Mock(return_value=scheduler)
        config = OmegaConf.create({'training': {
            'learning_rate': 1e-3, 'wan_learning_rate': 1e-4, 'weight_decay': .01}})
        actual, optimizer, actual_scheduler = self.namespace['create_model_and_optimizer'](
            config, model_factory=factory)
        self.assertIs(actual, model)
        self.assertIs(actual_scheduler, scheduler)
        factory.assert_called_once_with(config)
        self.assertEqual([g['lr'] for g in optimizer.param_groups], [1e-3, 1e-4])
        optimized = [p for group in optimizer.param_groups for p in group['params']]
        self.assertEqual({id(p) for p in optimized},
                         {id(p) for p in model.parameters() if p.requires_grad})
        self.assertEqual(len(optimized), len({id(p) for p in optimized}))


if __name__ == '__main__':
    unittest.main()
