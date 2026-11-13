"""Focused independent numerical and actual action-expert integration checks."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import nn
from torch.nn.utils import parametrize

from motus_actions.spline import AffineScale, decode, extract_local, fit_episode, local_curve, video_indices
from utils.lora import FusedQKVLoRA, LoRALinear, merge_lora


class SplineTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(240)
        self.actions = np.column_stack((np.sin(t / 20), np.cos(t / 30)))
        self.curve, self.error = fit_episode(self.actions)

    def test_fit_independent_basis(self):
        t = np.linspace(0, 239, 301)
        basis = BSpline.design_matrix(t, self.curve.t, 3).toarray()
        np.testing.assert_allclose(basis @ self.curve.c, self.curve(t), atol=1e-12)
        self.assertLessEqual(np.max(np.abs(self.curve(np.arange(240)) - self.actions)), .01)
        # Adaptive knots must respond to motion, not a fixed uniform grid.
        self.assertGreater(len(np.unique(np.diff(np.unique(self.curve.t)))), 1)

    def test_curve_against_analytic_bernstein_polynomial(self):
        # A single clamped cubic span is analytically a Bezier curve; this
        # expected result uses no SciPy basis or evaluation implementation.
        controls=np.array([[0.,2.],[1.,-1.],[3.,4.],[-2.,1.]])
        curve=BSpline(np.r_[np.zeros(4),np.ones(4)],controls,3)
        times=np.linspace(0,1,41)
        u=times[:,None]
        expected=(1-u)**3*controls[0]+3*u*(1-u)**2*controls[1]+3*u*u*(1-u)*controls[2]+u**3*controls[3]
        np.testing.assert_allclose(decode(extract_local(curve,0),times),expected,atol=2e-7)

    def test_local_global_time_origin_and_tail(self):
        for observation in (0, 1, 30, 80, 170, 237, 238):
            local = extract_local(self.curve, observation)
            curve = local_curve(local)
            ts = np.linspace(max(0, curve.t[3]), curve.t[-4], 25)
            np.testing.assert_allclose(decode(local, ts), self.curve(ts + observation), atol=1e-6)
            self.assertEqual(local.shape, (36, 3))

    def test_parameter_round_trip_constant_channels(self):
        scale = AffineScale([-10, 2, 0], [100, 2, 5])
        local = extract_local(self.curve, 30)
        np.testing.assert_allclose(scale.decode(scale.encode(local)), local, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            scale.decode([[np.nan, 0, 1]])

    def test_video_mapping_support_and_padding(self):
        # Manufactured nonuniform fitted knots make the expected indices independent.
        knots = np.r_[np.zeros(4), np.arange(1, 33) ** 1.2, np.repeat(70., 4)]
        curve = BSpline(knots, np.arange((len(knots) - 4) * 2).reshape(-1, 2), 3)
        local = extract_local(curve, 0)
        expected = np.floor(knots[[4, 8, 12, 16, 20, 24, 28, 32]] + .5).astype(int)
        np.testing.assert_array_equal(video_indices(local, 0, 100), expected)
        self.assertGreater(expected[0], 0)
        with self.assertRaisesRegex(ValueError, "padded|duplicate"):
            video_indices(extract_local(self.curve, 238), 238, 240)

    def test_invalid_predictions(self):
        local = extract_local(self.curve, 30)
        invalid = local.copy(); invalid[10, 0] = invalid[9, 0] - 1
        with self.assertRaisesRegex(ValueError, "ordering"):
            decode(invalid, [0, 1])
        invalid = local.copy(); invalid[:, 0] = 0
        with self.assertRaisesRegex(ValueError, "degenerate"):
            decode(invalid, [0, 1])
        invalid = local.copy(); invalid[0, 1] = np.inf
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            decode(invalid, [0, 1])
        with self.assertRaisesRegex(ValueError, "coverage"):
            decode(local, [0, 1000])

    def test_reference_running_max_counts_changes_without_reordering_controls(self):
        from motus_actions.spline import repair_prediction_knots
        local=extract_local(self.curve,30)
        invalid=local.copy()
        invalid[10,0]=invalid[9,0]-1
        repaired,count=repair_prediction_knots(invalid)
        self.assertEqual(count,1)
        self.assertEqual(repaired[10,0],repaired[9,0])
        np.testing.assert_array_equal(repaired[:,1:],invalid[:,1:])
        self.assertTrue(np.all(np.diff(repaired[:,0])>=0))
        np.testing.assert_array_equal(invalid[10],np.r_[local[9,0]-1,local[10,1:]])
        self.assertTrue(np.isfinite(decode(repaired,[0.,1.])).all())

    def test_native_controller_validation_and_counted_gripper_clip(self):
        from motus_actions.spline import controller_commands
        actions=np.zeros((2,14))
        actions[0,6]=1.01; actions[1,13]=-.01
        commands,repairs=controller_commands(actions,np.tile([-2.,2.],(12,1)))
        self.assertEqual(repairs,2)
        self.assertEqual(commands[0,6],1.)
        self.assertEqual(commands[1,13],0.)
        self.assertEqual(actions[0,6],1.01)  # inputs immutable
        actions[0,0]=3.
        with self.assertRaisesRegex(ValueError,'native arm limits'):
            controller_commands(actions,np.tile([-2.,2.],(12,1)))


class AdapterTests(unittest.TestCase):
    def test_configured_rank_256_reaches_linear_and_fused_adapters(self):
        from utils.lora import configure_lora
        model=nn.Module()
        model.video_model=nn.Module(); model.video_model.wan_model=nn.Module()
        block=nn.Module()
        for name in ('self_attn','cross_attn'):
            attention=nn.Module()
            for projection in ('q','k','v','o'): setattr(attention,projection,nn.Linear(8,8))
            setattr(block,name,attention)
        model.video_model.wan_model.blocks=nn.ModuleList([block])
        model.action_expert=nn.Module()
        action_block=nn.Module()
        action_block.wan_action_qkv=nn.Parameter(torch.randn(3,2,8,4))
        action_block.wan_action_o=nn.Linear(8,8)
        model.action_expert.blocks=nn.ModuleList([action_block])
        model.action_expert.input_encoder=nn.Module()
        model.action_expert.input_encoder.action_encoder=nn.Linear(8,8)
        model.action_expert.decoder=nn.Module()
        model.action_expert.decoder.action_head=nn.Linear(8,8)
        summary=configure_lora(model,rank=256,alpha=32)
        self.assertEqual(summary['rank'],256)
        self.assertEqual(summary['scale'],.125)
        self.assertEqual(block.self_attn.q.a.shape,(256,8))
        self.assertEqual(action_block.parametrizations.wan_action_qkv[0].a.shape,(3,256,8))
        x=torch.randn(2,8)
        torch.testing.assert_close(block.self_attn.q(x),block.self_attn.q.base(x),rtol=0,atol=0)

    def test_fused_qkv_matches_independent_linear_projections(self):
        torch.manual_seed(4)
        weight = torch.randn(3, 2, 8, 4)
        adapter = FusedQKVLoRA(weight, rank=2, alpha=4)
        with torch.no_grad():
            adapter.b.normal_()
        inputs = torch.randn(2, 5, 8, requires_grad=True)
        actual = torch.einsum('bti,knid->kbtnd', inputs, adapter(weight))
        for k in range(3):
            base = weight[k].permute(0, 2, 1).reshape(8, 8)
            expected = inputs @ base.T + (inputs @ adapter.a[k].T @ adapter.b[k].T) * 2
            torch.testing.assert_close(actual[k].flatten(-2), expected)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(adapter.a.grad).all())
        self.assertGreater(adapter.b.grad.abs().sum().item(), 0)

    def test_checkpoint_C_adapts_both_registered_expert_aliases(self):
        from types import SimpleNamespace
        from models.action_expert import ActionExpert, ActionExpertConfig
        from utils.lora import load_initial_weights
        def make(rows,width,representation):
            model=nn.Module()
            model.config=SimpleNamespace(action_representation=representation)
            model.action_expert=ActionExpert(ActionExpertConfig(dim=128,ffn_dim=256,num_layers=1,
                state_dim=14,action_dim=width,chunk_size=rows+1),{'dim':128,'num_heads':1,'head_dim':128})
            model.action_module=nn.Module()
            model.action_module.action_expert=model.action_expert
            return model
        source,target=make(16,14,'dense'),make(36,15,'spline')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'initial.pt'
            torch.save({'module':source.state_dict()},checkpoint)
            report=load_initial_weights(target,checkpoint,Path(directory)/'report.json')
            self.assertFalse(report['shape_mismatched'])
            self.assertEqual(len(report['missing']),2)
            self.assertEqual(len(report['adapted']),8)
            weight=target.action_expert.input_encoder.action_encoder[0].weight
            torch.testing.assert_close(weight[:,1:],source.action_expert.input_encoder.action_encoder[0].weight)
            torch.testing.assert_close(weight[:,0],torch.zeros(128))
            self.assertIs(target.action_module.action_expert,target.action_expert)

    def test_linear_and_fused_merge(self):
        layer = LoRALinear(nn.Linear(8, 8), rank=2)
        with torch.no_grad(): layer.b.normal_()
        inputs = torch.randn(4, 8)
        torch.testing.assert_close(layer(inputs), layer.merged()(inputs), atol=1e-5, rtol=1e-5)
        block = nn.Module()
        block.wan_action_qkv = nn.Parameter(torch.randn(3, 2, 8, 4))
        parametrize.register_parametrization(block, 'wan_action_qkv', FusedQKVLoRA(block.wan_action_qkv, rank=2))
        with torch.no_grad(): block.parametrizations.wan_action_qkv[0].b.normal_()
        expected = block.wan_action_qkv.detach().clone()
        merge_lora(block)
        torch.testing.assert_close(expected, block.wan_action_qkv)

    def test_save_load_resume_optimizer_scheduler_and_rng(self):
        import random
        from accelerate import Accelerator
        torch.manual_seed(5)
        model = LoRALinear(nn.Linear(8, 8), rank=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
        accelerator = Accelerator(cpu=True)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        x = torch.randn(2, 8)
        accelerator.backward(model(x).square().mean())
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        with tempfile.TemporaryDirectory() as directory:
            accelerator.save_state(directory)
            expected = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_scheduler = copy.deepcopy(scheduler.state_dict())
            expected_python = random.random()
            expected_numpy = np.random.random(4)
            expected_rng = torch.randn(4)
            expected_lr = optimizer.param_groups[0]['lr']
            accelerator.backward(model(x).square().mean())
            optimizer.step(); scheduler.step()
            accelerator.load_state(directory)
            for name, value in expected.items(): torch.testing.assert_close(value, model.state_dict()[name])
            torch.testing.assert_close(torch.randn(4), expected_rng)
            self.assertEqual(expected_lr, optimizer.param_groups[0]['lr'])
            self.assertEqual(random.random(), expected_python)
            np.testing.assert_array_equal(np.random.random(4), expected_numpy)
            self.assertEqual(scheduler.state_dict(), expected_scheduler)
            actual_optimizer = optimizer.state_dict()
            self.assertEqual(actual_optimizer['param_groups'], expected_optimizer['param_groups'])
            for key, state in expected_optimizer['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(actual_optimizer['state'][key][name], value, rtol=0, atol=0)

    def test_actual_action_expert_shapes_and_gradient(self):
        from models.motus import MotusConfig
        from models.action_expert import ActionExpert, ActionExpertConfig
        for representation, rows, width in [('dense', 16, 14), ('spline', 36, 15)]:
            config = MotusConfig(action_representation=representation, num_video_frames=8, video_action_freq_ratio=2)
            self.assertEqual((config.action_chunk_size, config.action_token_dim), (rows, width))
            self.assertEqual(config.action_state_dim, 14)
            expert = ActionExpert(ActionExpertConfig(dim=128, ffn_dim=256, num_layers=1,
                state_dim=14, action_dim=width, chunk_size=rows + 1), {'dim': 128, 'num_heads': 1, 'head_dim': 128})
            state, actions = torch.randn(2, 1, 14), torch.randn(2, rows, width)
            tokens = expert.input_encoder(state, actions, expert.registers.expand(2, -1, -1))
            self.assertEqual(tokens.shape, (2, rows + 5, 128))
            pred = expert.decoder(tokens, torch.randn(2, rows + 5, 128))[:, 1:-4]
            self.assertEqual(pred.shape, actions.shape)
            loss = (pred - actions).square().mean()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(expert.input_encoder.action_encoder[0].weight.grad)
            self.assertGreater(expert.decoder.action_head[-1].weight.grad.abs().sum().item(), 0)


class PipelineTests(unittest.TestCase):
    def test_accumulation_counts_optimizer_updates(self):
        import importlib.util
        import sys
        from accelerate import Accelerator
        from omegaconf import OmegaConf
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'train'))
        spec = importlib.util.spec_from_file_location('motus_existing_trainer', root / 'train/train.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class TinyFlow(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.))
            def training_step(self, actions, **kwargs):
                loss = (self.weight - actions.float().mean()).square()
                return dict(total_loss=loss, video_loss=loss * 0, action_loss=loss)

        batch = dict(first_frame=torch.zeros(1,3,4,4), video_frames=torch.zeros(1,8,3,4,4),
                     action_sequence=torch.zeros(1,16,14), initial_state=torch.zeros(1,14),
                     language_embedding=None, vlm_inputs=None)
        from torch.utils.data import DataLoader
        for representation, rows, width in [('A',16,14),('C',36,15)]:
            accelerator = module.create_accelerator(16, cpu=True)
            model = TinyFlow()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
            sample = {k: v.squeeze(0) if torch.is_tensor(v) else v for k,v in batch.items()}
            sample['action_sequence'] = torch.zeros(rows,width)
            from data.dataset import collate_fn
            loader = DataLoader([sample]*450, batch_size=1, collate_fn=collate_fn,
                drop_last=True, generator=torch.Generator().manual_seed(0))
            model,optimizer,loader,scheduler=module.prepare_pilot_training(accelerator,model,optimizer,loader,scheduler)
            config = OmegaConf.create({'training': {'batch_size':1, 'grad_clip_norm':10., 'lora':True}})
            with tempfile.TemporaryDirectory() as directory:
                trainer = module.UniDiffuserTrainer(model, loader, optimizer=optimizer, scheduler=scheduler,
                    device='cpu', checkpoint_dir=directory, log_interval=100, report_to=[],
                    accelerator=accelerator, config=config)
                trainer.train(max_steps=57, val_interval=100)
                self.assertEqual(trainer.epoch,2)
                self.assertEqual(trainer.batches_in_epoch,12)
                self.assertEqual([r['examples'] for r in trainer.update_records],[16]*57)
                self.assertEqual([r['microbatches'] for r in trainer.update_records],[16]*57)
                self.assertEqual(int(next(iter(optimizer.state_dict()['state'].values()))['step']),57)
                self.assertEqual(scheduler.state_dict()['last_epoch'],57)
                self.assertTrue((Path(directory)/'checkpoint_step_57/trainer_state.json').exists())
                print(f'{representation}: 912 microbatches, 2 epoch boundaries, 57 optimizer/scheduler steps, all updates=16 examples')

    def test_experiment_training_and_inference_VLM_inputs_match(self):
        from PIL import Image
        from transformers import AutoProcessor
        from utils.vlm_utils import preprocess_vlm_messages
        from inference.robotwin.Motus.deploy_policy import MotusPolicy
        root=Path(__file__).resolve().parents[1]
        processor=AutoProcessor.from_pretrained(root/'.runtime/model-metadata/Qwen3-VL-2B-Instruct')
        policy=MotusPolicy.__new__(MotusPolicy)
        policy.experiment_lora=True
        policy.vlm_processor=processor
        policy.device='cpu'
        image=Image.fromarray(np.full((384,320,3),128,dtype=np.uint8))
        expected=preprocess_vlm_messages('Place the can in the basket.',image,processor)
        actual=policy._preprocess_vlm_messages('Place the can in the basket.',image)
        self.assertEqual(set(actual),set(expected))
        for name in actual:
            torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)

    def test_real_manifest_A_C_share_current_observation(self):
        import random
        from omegaconf import OmegaConf
        from data.dataset import create_dataset
        root = Path(__file__).resolve().parents[1]
        config = OmegaConf.load(root/'configs/robotwin_ac.yaml')
        manifest = Path(config.dataset.episode_manifest)
        if not manifest.exists() or not (manifest.parent/'language').exists():
            self.skipTest('real cache not prepared')
        samples = []
        for representation in ('dense','spline'):
            config.common.action_representation = representation
            dataset = create_dataset(config)
            random.seed(13)
            samples.append(dataset[0])
        torch.testing.assert_close(samples[0]['first_frame'], samples[1]['first_frame'])
        torch.testing.assert_close(samples[0]['initial_state'], samples[1]['initial_state'])
        self.assertEqual(tuple(samples[0]['action_sequence'].shape), (16,14))
        self.assertEqual(tuple(samples[1]['action_sequence'].shape), (36,15))
        self.assertEqual(tuple(samples[1]['video_frames'].shape), (8,3,384,320))
        self.assertTrue(torch.isfinite(samples[1]['action_sequence']).all())


if __name__ == '__main__':
    unittest.main()
