"""CPU component contracts, NOT full MOTUS or real-demonstration learning.

Load actual component definitions by AST to avoid allocating/importing unavailable
5B/VLM dependencies. The full model, GPU kernels and VAE remain remote checks.
"""
import ast
import dataclasses
from functools import lru_cache
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
import types
import typing
import unittest
import uuid

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.device import amp_autocast, disable_autocast
from utils.research import (configure_adapters, trainable_summary, gradient_summary,
                            load_compatible_base, save_research_bundle, load_research_bundle, masked_video_loss)


def definitions(path, names, namespace):
    source = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    body = [n for n in source.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ROOT / path), 'exec'), namespace)


def component_namespace():
    module = types.ModuleType('research_component_fixture')
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update(torch=torch, nn=nn, np=np, math=math, re=re, dataclass=dataclasses.dataclass,
              amp_autocast=amp_autocast, disable_autocast=disable_autocast, lru_cache=lru_cache,
              logger=logging.getLogger('component-test'), **{k: getattr(typing, k) for k in ['List', 'Dict', 'Any', 'Tuple', 'Optional']})
    definitions('bak/wan/modules/model.py', {'sinusoidal_embedding_1d', 'rope_params', 'rope_apply', 'WanRMSNorm', 'WanLayerNorm', 'WanSelfAttention'}, ns)
    # CPU test fallback for the unchanged unmasked FlashAttention branch only.
    # Masked branch executes the actual newly integrated native SDPA code.
    def flash(q, k, v, **kwargs):
        dtype = v.dtype
        return torch.nn.functional.scaled_dot_product_attention(q.to(dtype).transpose(1,2), k.to(dtype).transpose(1,2), v.transpose(1,2)).transpose(1,2)
    ns['flash_attention'] = flash
    definitions('models/action_expert.py', {'get_1d_sincos_pos_embed_from_grid', 'ActionExpertConfig', 'StateActionEncoder', 'ActionEncoder', 'ActionExpertBlock', 'ActionDecoder', 'ActionExpert'}, ns)
    definitions('models/und_expert.py', {'UndExpertConfig', 'UndExpertBlock', 'UndExpert'}, ns)
    definitions('models/motus.py', {'MotusConfig', 'VideoModule'}, ns)
    return ns


class Harness(nn.Module):
    """Two small actual expert/attention layers, without VAE or VLM weights."""
    def __init__(self, representation='spline'):
        super().__init__()
        ns = component_namespace()
        self.config = ns['MotusConfig'](action_representation=representation, action_dim=15 if representation == 'spline' else 14,
                                       num_video_frames=8, video_action_freq_ratio=2, action_expert_dim=16, num_layers=2)
        wan = dict(dim=12, num_heads=1, head_dim=12)
        self.action_expert = ns['ActionExpert'](ns['ActionExpertConfig'](dim=16, ffn_dim=32, num_layers=2, state_dim=14,
                  action_dim=self.config.action_dim, chunk_size=self.config.action_chunk_size + 1, num_registers=4), wan)
        self.und_expert = ns['UndExpert'](ns['UndExpertConfig'](dim=8, ffn_dim=16, num_layers=2, vlm_input_dim=8), wan)
        self.video_model = nn.Module()
        self.video_model.wan_model = nn.Module()
        video = self.video_model.wan_model
        video.num_heads = 1
        video.register_buffer('freqs', ns['rope_params'](32, 12))
        blocks = []
        for _ in range(2):
            block = nn.Module()
            block.norm1 = ns['WanLayerNorm'](12)
            block.self_attn = ns['WanSelfAttention'](12, 1)
            block.ffn = nn.Sequential(nn.Linear(12, 24), nn.GELU(), nn.Linear(24, 12))
            blocks.append(block)
        video.blocks = nn.ModuleList(blocks)
        self.vlm_model = nn.Linear(3, 3)  # explicitly frozen, never called
        self.video_module = ns['VideoModule'](self.video_model, torch.float32, torch.device('cpu'), torch.tensor([[3,1,1]]))

    def forward(self, state, actions, video, und):
        tokens = self.action_expert.input_encoder(state[:, None], actions, self.action_expert.registers.expand(len(state), -1, -1))
        for i in range(2):
            vmod = [torch.zeros(len(state), 3, 1, 12), torch.zeros(len(state), 3, 1, 12), torch.ones(len(state), 3, 1, 12)]
            amod = [torch.zeros(len(state), tokens.shape[1], 1, 16), torch.zeros(len(state), tokens.shape[1], 1, 16), torch.ones(len(state), tokens.shape[1], 1, 16)]
            video, tokens, und = self.video_module.process_joint_attention(video, tokens, vmod, amod, i,
                self.action_expert.blocks[i], und, self.und_expert.blocks[i], video_latent_valid=torch.tensor([[True, True, False]]).expand(len(state), -1))
            video = video + self.video_model.wan_model.blocks[i].ffn(video)
            tokens = tokens + self.action_expert.blocks[i].ffn(tokens)
            und = und + self.und_expert.blocks[i].ffn(und)
        output = self.action_expert.decoder(tokens, torch.zeros_like(tokens))[:, 1:-4]
        return output, video, und


def fixture_directory():
    path = ROOT / 'outputs' / 'stage3-components' / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def build_bundle_model(directory):
    torch.manual_seed(0)
    model = Harness('spline')
    model.research_base = load_compatible_base(model, directory / 'base.pt')
    torch.manual_seed(0)
    configure_adapters(model, dict(rank=2, alpha=4., dropout=0.))
    model.research_contract = dict(schema='CPU-COMPONENT-ONLY', representation='spline', scope='synthetic diagnostic-only')
    return model


class ComponentTests(unittest.TestCase):
    def test_generation_entry_never_reads_future_frames(self):
        ns = dict(torch=torch, Dict=typing.Dict, Tuple=typing.Tuple)
        definitions('train/sample.py', {'inference_sample'}, ns)
        class Endpoint(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
            def inference_step(self, first_frame, state, num_inference_steps, language_embeddings, vlm_inputs):
                return first_frame, state
        model = Endpoint()
        # Deliberately omit video_frames and action_sequence entirely.
        batch = dict(first_frame=torch.zeros(1,3,2,2), initial_state=torch.ones(1,14),
                     language_embedding=torch.zeros(1,2,4), vlm_inputs=None)
        config = types.SimpleNamespace(model=types.SimpleNamespace(inference=types.SimpleNamespace(num_inference_timesteps=10)))
        frame, state = ns['inference_sample'](model, batch, config)
        torch.testing.assert_close(frame, batch['first_frame'])
        torch.testing.assert_close(state, batch['initial_state'])

    def test_existing_loader_and_collate_keep_current_conditioning_identical(self):
        from data.robotwin2.adaptive_spline import fit_episode, normalize, extract_local, select_future_frames
        directory = fixture_directory()
        t = np.linspace(0, 1, 120)
        actions = np.column_stack([np.sin((j+1)*t)*.3 for j in range(14)]).astype(np.float32).astype(np.float64)
        actions[:,6] = (t>.3)&(t<.75)
        norm = dict(low=actions.min(0).tolist(), high=actions.max(0).tolist())
        spline, _ = fit_episode(normalize(actions, norm))
        scaling = dict(scope='diagnostic-only', units='source_action_step', offset=0., scale=100.)
        packet = extract_local(spline, 0, scaling)
        selection = select_future_frames(packet, np.arange(120), np.arange(120))
        qpos = directory / 'source.pt'
        torch.save(torch.tensor(actions, dtype=torch.float32), qpos)
        ns = dict(__name__='data.robotwin2.component_loader', __package__='data.robotwin2',
                  torch=torch, data=torch.utils.data, np=np, Path=Path,
                  **{k: getattr(typing,k) for k in ['Dict','List','Any','Optional','Tuple']})
        ns['load_video_frames'] = lambda path, indices, size: torch.stack([torch.full((3,2,2), i / 120) for i in indices])
        ns['get_video_frame_count'] = lambda path: 120
        ns['tensor_to_pil'] = lambda frame: frame
        def preprocess(text, current, processor):
            self.assertEqual(float(current.max()), 0.)
            return dict(input_ids=torch.ones(1,2,dtype=torch.long), attention_mask=torch.ones(1,2,dtype=torch.long))
        ns['preprocess_vlm_messages'] = preprocess
        definitions('data/robotwin2/robotwin_agilex_dataset.py', {'RobotWinTaskDataset'}, ns)
        ns.update({k: getattr(typing,k) for k in ['Dict','List','Any','Optional']})
        definitions('data/dataset.py', {'_process_vlm_inputs_batch','_process_language_embeddings_batch','collate_fn'}, ns)
        outputs = []
        window = dict(qpos_path=str(qpos), video_path='synthetic-decoder-stub', lang_path='unused', task_name='place_can_basket',
                      episode_name='0', data_mode='clean', observation_index=0, cache_path='synthetic', identity='component-only',
                      _actions=actions, _dense_indices=np.arange(3,49,3), _packet=packet, _selection=selection,
                      _timeline=dict(action_times=np.arange(120),camera_times=np.arange(120)))
        for representation in ['dense','spline']:
            dataset = ns['RobotWinTaskDataset'].__new__(ns['RobotWinTaskDataset'])
            dataset.research_windows, dataset.research_contract = [window], dict(representation=representation)
            dataset.vlm_processor, dataset.video_size = object(), (2,2)
            dataset._load_language_embedding = lambda path, index: (torch.zeros(2,4), index)
            dataset._load_text_instruction = lambda *a, **kw: 'place can basket'
            outputs.append(ns['collate_fn']([dataset._research_sample(0)]))
        torch.testing.assert_close(outputs[0]['initial_state'], outputs[1]['initial_state'], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0]['first_frame'], outputs[1]['first_frame'], rtol=0, atol=0)
        torch.testing.assert_close(outputs[0]['vlm_inputs']['input_ids'], outputs[1]['vlm_inputs']['input_ids'])
        self.assertEqual(tuple(outputs[0]['action_sequence'].shape), (1,16,14))
        self.assertEqual(tuple(outputs[1]['action_sequence'].shape), (1,36,15))
        self.assertFalse(torch.equal(outputs[0]['video_frames'], outputs[1]['video_frames']))
        self.assertEqual(tuple(outputs[1]['video_latent_valid'].shape), (1,3))

    def test_actual_flow_assignments_resample_noise_time_and_preserve_sign(self):
        tree = ast.parse((ROOT / 'models/motus.py').read_text())
        model_class = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Motus')
        step = next(n for n in model_class.body if isinstance(n,ast.FunctionDef) and n.name=='training_step')
        names = {'timestep_id_action','action_t_embed','sigma_action','action_noise','noisy_actions','action_target'}
        body = [n for n in step.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names]
        code = compile(ast.Module(body=body, type_ignores=[]), 'actual-MOTUS-action-flow', 'exec')
        scheduler = types.SimpleNamespace(num_train_timesteps=1000, timesteps=torch.arange(1000), sigmas=torch.linspace(0,1,1000))
        owner = types.SimpleNamespace(fm_train_scheduler_action=scheduler, device=torch.device('cpu'), action_dtype=torch.float32)
        actions = torch.randn(32,36,15)
        previous = None
        for _ in range(2):
            ns = dict(torch=torch, B=32, self=owner, actions=actions)
            exec(code,ns)
            torch.testing.assert_close(ns['noisy_actions'], actions*(1-ns['sigma_action'])+ns['action_noise']*ns['sigma_action'])
            torch.testing.assert_close(ns['action_target'], ns['action_noise']-actions)
            torch.testing.assert_close(ns['action_noise'] - ns['action_target'], actions)
            self.assertGreater(ns['timestep_id_action'].unique().numel(), 1)
            if previous is not None:
                self.assertFalse(torch.equal(previous,ns['action_noise']))
            previous=ns['action_noise']

    def test_A_C_shapes_scope_optimizer_and_actual_gradients(self):
        reports = {}
        for representation, rows, width in [('dense', 16, 14), ('spline', 36, 15)]:
            torch.manual_seed(0)
            model = Harness(representation)
            configure_adapters(model, dict(rank=2, alpha=4., dropout=0.))
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
            summary = trainable_summary(model, opt)
            inputs = [torch.randn(1,14), torch.randn(1,rows,width), torch.randn(1,3,12), torch.randn(1,2,8)]
            target = torch.randn(1,rows,width)
            for _ in range(3):
                opt.zero_grad()
                output, video, und = model(*inputs)
                self.assertEqual(output.shape, (1, rows, width))
                (torch.nn.functional.mse_loss(output, target) + video.square().mean()).backward()
                opt.step()
            gradients = gradient_summary(model)
            self.assertTrue(all(g is not None and g['finite'] for g in gradients.values()))
            self.assertTrue(all(g['norm'] > 0 for g in gradients.values()))
            self.assertFalse(any(p.requires_grad for p in model.vlm_model.parameters()))
            self.assertFalse(model.action_expert.blocks[0].wan_action_qkv.requires_grad)
            self.assertFalse(any(p.requires_grad for p in model.und_expert.blocks[-1].parameters()))
            expected_tokens = rows + 1 + 4
            self.assertEqual(model.action_expert.input_encoder(inputs[0][:,None], inputs[1], model.action_expert.registers).shape[1], expected_tokens)
            reports[representation] = dict(summary=summary, action_branch_tokens=expected_tokens, gradients=gradients)
        self.assertEqual(reports['dense']['summary']['adapter_targets'], reports['spline']['summary']['adapter_targets'])
        out = fixture_directory()
        (out / 'trainable-and-gradient-summary.json').write_text(json.dumps(reports, indent=2))
        print('CPU component trainables/gradients:', out)

    def test_masked_keys_cannot_affect_action_or_understanding(self):
        ns = component_namespace()
        attention = ns['WanSelfAttention'](12, 1)
        qkv = [torch.randn(1, 2, 1, 12) for _ in range(6)]
        video = torch.randn(1, 3, 12)
        mask = torch.tensor([[True, True, False, True, True, True, True]])
        args = dict(action_q=qkv[0], action_k=qkv[1], action_v=qkv[2], und_q=qkv[3], und_k=qkv[4], und_v=qkv[5], joint_key_valid=mask)
        freqs = ns['rope_params'](32, 12)
        first = attention(video, torch.tensor([7]), torch.tensor([[3,1,1]]), freqs, **args)
        changed = video.clone()
        changed[:,2] += 1000
        second = attention(changed, torch.tensor([7]), torch.tensor([[3,1,1]]), freqs, **args)
        torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)
        torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)

    def test_masked_loss_preserves_dense_scale_and_excludes_padding(self):
        pred = torch.randn(2, 3, 3, 2, 2, requires_grad=True)
        target = torch.randn_like(pred)
        keep = torch.ones(2,3,dtype=torch.bool)
        reference = (pred - target).clone()
        reference[:,:,0] = 0
        torch.testing.assert_close(masked_video_loss(pred, target, keep), reference.square().mean())
        masked_video_loss(pred, target, torch.tensor([[True,True,False]]).expand(2,-1)).backward()
        self.assertTrue(torch.all(pred.grad[:,:,0] == 0))
        self.assertTrue(torch.all(pred.grad[:,:,2] == 0))
        with self.assertRaises(ValueError):
            masked_video_loss(pred, target, torch.tensor([[True,False,False]]).expand(2,-1))

    def test_serialized_adapters_projections_fresh_process(self):
        directory = fixture_directory()
        torch.manual_seed(0)
        model = Harness('spline')
        torch.save(model.state_dict(), directory / 'base.pt')
        model = build_bundle_model(directory)
        inputs = [torch.randn(1,14), torch.randn(1,36,15), torch.randn(1,3,12), torch.randn(1,2,8)]
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
        for _ in range(2):
            opt.zero_grad()
            y, v, u = model(*inputs)
            (y.square().mean() + v.square().mean() + y.mean()).backward()
            opt.step()
        save_research_bundle(model, directory, dict(component_test=True))
        torch.save(inputs, directory / 'inputs.pt')
        np.save(directory / 'expected.npy', model(*inputs)[0].detach().numpy())
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fresh-reload', str(directory)],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads((directory / 'fresh-reload.json').read_text())
        self.assertEqual(report['maximum_difference'], 0.)
        bad = build_bundle_model(directory)
        bad.research_contract['representation'] = 'dense'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            load_research_bundle(bad, directory)
        print('Fresh-process CPU component checkpoint passed:', directory)

    def test_accumulation_across_loader_boundaries_uses_actual_trainer(self):
        from accelerate import Accelerator
        from accelerate.utils import GradientAccumulationPlugin
        class Config(dict):
            __getattr__ = dict.__getitem__
        ns = dict(torch=torch, Dict=typing.Dict, Any=typing.Any, Optional=typing.Optional,
                  OmegaConf=object, Motus=nn.Module, SummaryWriter=object, Accelerator=object,
                  DataLoader=torch.utils.data.DataLoader, Path=Path, logger=logging.getLogger('test'))
        definitions('train/train.py', {'UniDiffuserTrainer'}, ns)
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(1))
            def forward(self, first_frame, actions, **kwargs):
                loss = (self.weight * first_frame - actions).square().mean()
                return dict(total_loss=loss, action_loss=loss, video_loss=loss*0)
        accelerator = Accelerator(cpu=True, gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=4, sync_with_dataloader=False))
        model = Toy()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        loader = torch.utils.data.DataLoader([0,1,2], batch_size=1)
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
        trainer = ns['UniDiffuserTrainer'].__new__(ns['UniDiffuserTrainer'])
        trainer.model, trainer.optimizer, trainer.accelerator = model, optimizer, accelerator
        trainer.device, trainer.dtype, trainer.scheduler = torch.device('cpu'), torch.float32, None
        class Counter:
            count = 0
            def step(self):
                self.count += 1
        trainer.scheduler = Counter()
        trainer.config = Config(training=Config(grad_clip_norm=100.))
        batch = dict(first_frame=torch.ones(1,1), video_frames=torch.zeros(1,8,3,1,1),
                     action_sequence=torch.full((1,1),2.), vlm_inputs=None, language_embedding=None)
        iterator = iter(loader)
        updates = []
        optimizer.zero_grad()
        for i in range(8):
            try:
                next(iterator)
            except StopIteration:
                iterator = iter(loader)
                next(iterator)
            if trainer.train_step(batch)['_optimizer_step']:
                updates.append(i+1)
        self.assertEqual(updates, [4,8])
        self.assertEqual(trainer.scheduler.count, 2)
        torch.testing.assert_close(accelerator.unwrap_model(model).weight.detach(), torch.tensor([.72]))


if __name__ == '__main__':
    torch.set_num_threads(1)
    if len(sys.argv) > 1 and sys.argv[1] == '--fresh-reload':
        directory = Path(sys.argv[2])
        model = build_bundle_model(directory)
        load_research_bundle(model, directory)
        inputs = torch.load(directory / 'inputs.pt', weights_only=True)
        actual = model(*inputs)[0].detach().numpy()
        difference = float(np.max(np.abs(actual - np.load(directory / 'expected.npy'))))
        (directory / 'fresh-reload.json').write_text(json.dumps(dict(scope='CPU components only', maximum_difference=difference)))
    else:
        unittest.main()
