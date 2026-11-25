"""A/C adapter and actual RoboTwin rollout function, without GPU/native scene setup."""
import ast
from collections.abc import Mapping, Sequence
import copy
import json
import logging
import multiprocessing as mp
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from experiments.ac_spline_pilot.rollout_policy import RolloutPolicy, ROBOTWIN_ROOT, load_policy
from experiments.ac_spline_pilot.simulator import evaluation_cases, evaluate_cases


def rollout_api():
    source = ROBOTWIN_ROOT / 'scripts/eval_policy_xpolicylab.py'
    names = {'run_one_batch_episode', 'allocate_batch_episode_id', 'prepare_policy_case',
             'reset_policy', 'notify_trial_end', 'is_episode_end', 'robotwin_obs_to_xpolicylab',
             'normalize_action_chunk', 'xpolicylab_action_to_robotwin', 'normalize_robotwin_action_type',
             'convert_vision', 'convert_camera', 'convert_state'}
    tree = ast.parse(source.read_text())
    ns = dict(np=np, Any=object, Mapping=Mapping, Sequence=Sequence, time=time,
        traceback=traceback, subprocess=subprocess, UnStableError=RuntimeError,
        build_instruction=Mock(side_effect=AssertionError('Use the fixed manifest instruction')),
        CAMERA_NAME_MAP={'head_camera':'cam_head','left_camera':'cam_left_wrist','right_camera':'cam_right_wrist'})
    exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), str(source), 'exec'), ns)
    return SimpleNamespace(**{name: ns[name] for name in names})


class Model(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.prediction = torch.as_tensor(parameters, dtype=torch.float32)[None]
    def inference_step(self, **kwargs):
        self.inputs = kwargs
        return None, self.prediction


def policy(spline=False):
    packet = np.column_stack((np.r_[np.zeros(4), np.arange(1.,29.), np.full(4,60.)], np.zeros((36,14))))
    if spline:
        packet[10,0] = packet[9,0] - 1
    model = Model(packet if spline else np.full((16,14),.25))
    config = SimpleNamespace(common=SimpleNamespace(video_height=24,video_width=16),
        model=SimpleNamespace(inference=SimpleNamespace(num_inference_timesteps=10)))
    contract = dict(representation='spline' if spline else 'dense',
        executed_prefix=dict(command_count=16,source_stride=3),
        knot_scaling=dict(scope='diagnostic-only',units='source_action_step',offset=0.,scale=1.),
        normalization=dict(low=[-1.]*14,high=[1.]*14))
    return RolloutPolicy(model,config,contract,{'place':torch.zeros(2,4096)},object())


def observation():
    cameras = {name:dict(rgb=np.full((16,16,3),color,np.uint8)) for name,color in
               [('head_camera',220),('left_camera',40),('right_camera',80)]}
    return dict(observation=cameras,joint_action=dict(left_arm=np.arange(6.),left_gripper=.1,
                     right_arm=np.arange(6.)+7,right_gripper=.9))


class Environment:
    def __init__(self):
        self.scene = SimpleNamespace(get_timestep=lambda:.004)
        self.render_freq=0
        self.total_commands=0
    def setup_demo(self, **kwargs):
        self.take_action_cnt=0
        self.action_physics_steps=0
        self.step_lim=4
        self.eval_success=False
        self.eval_video_path=None
        self.seed=kwargs['seed']
    def set_instruction(self,instruction): self.instruction=instruction
    def get_instruction(self): return self.instruction
    def get_obs(self): return observation()
    def take_action(self,command,action_type):
        assert action_type=='qpos'
        assert np.asarray(command).shape==(14,)
        self.total_commands+=1
        self.take_action_cnt+=1
        self.action_physics_steps+=3
        self.eval_success=self.take_action_cnt==2
    def close_env(self): pass


class SimulatorTests(unittest.TestCase):
    def test_adapter_uses_training_rgb_state_prompt_and_preserves_rng(self):
        p=policy()
        p.begin_episode(100000,'place')
        api=rollout_api()
        obs=api.robotwin_obs_to_xpolicylab(observation(),instruction='place')
        p.call('update_obs',obs=obs)
        before=torch.random.get_rng_state()
        with patch('utils.vlm_utils.preprocess_vlm_messages',return_value={}) as vlm:
            result=p.call('get_action')
        torch.testing.assert_close(torch.random.get_rng_state(),before)
        np.testing.assert_array_equal(result,np.full((16,14),.25))
        frame=p.model.inputs['first_frame'][0]
        self.assertAlmostEqual(float(frame[0,0,0]),220/255,places=6)
        self.assertAlmostEqual(float(frame[0,20,3]),40/255,places=6)
        self.assertAlmostEqual(float(frame[0,20,12]),80/255,places=6)
        np.testing.assert_allclose(p.model.inputs['state'][0],np.r_[np.arange(6.),.1,np.arange(6.)+7,.9])
        self.assertTrue(vlm.call_args.args[0].endswith('performing the following task: place'))
        self.assertTrue(p.calls[0]['decoded_valid'])

    def test_actual_rollout_loop_executes_predictions_and_records_timing(self):
        api=rollout_api()
        env=Environment()
        api.class_decorator=lambda name:env
        p=policy(spline=True)
        p.execution_prefix = dict(p.execution_prefix, row_budget=14)
        p.spline_diagnostics = True
        manifest=dict(cases=[dict(seed=100000,instruction='place'),dict(seed=100001,instruction='place')],instruction_type='seen')
        with tempfile.TemporaryDirectory() as tmp, patch('utils.vlm_utils.preprocess_vlm_messages',return_value={}):
            records=evaluate_cases(api,p,dict(task_name='place_can_basket',render_freq=0),manifest,Path(tmp))
            saved=[json.loads(line) for line in (Path(tmp)/'episodes.jsonl').read_text().splitlines()]
        self.assertEqual([r['seed'] for r in saved],[100000,100001])
        for record in records:
            self.assertTrue(record['success'])
            self.assertEqual(record['policy_calls'],1)
            self.assertEqual(record['executed_commands'],2)
            self.assertEqual(record['commands_per_call'],[2])
            self.assertEqual(record['action_physics_steps'],6)
            self.assertEqual(record['simulated_seconds'],.024)
            self.assertEqual(record['scene_timestep_seconds'],.004)
            self.assertGreaterEqual(record['execution_wall_seconds'],0.)
            self.assertTrue(record['calls'][0]['decoder_report']['repaired_knots'])
            report = record['calls'][0]['spline_diagnostics']
            self.assertEqual(report['episode_id'], record['episode_id'])
            self.assertEqual(report['policy_call_index'], 0)
            self.assertEqual(report['requested_interval_relative'], [3., 6.])
            self.assertTrue(report['coverage_valid'] and report['decoder_success'])
        self.assertEqual(env.total_commands,4)

    def test_invalid_prefix_is_a_failed_rollout_without_fabricated_commands(self):
        api=rollout_api()
        env=Environment()
        api.class_decorator=lambda name:env
        p=policy(spline=True)
        p.spline_diagnostics = True
        p.model.prediction[:,:,0]+=10
        manifest=dict(cases=[dict(seed=100000,instruction='place')],instruction_type='seen')
        with tempfile.TemporaryDirectory() as tmp, patch('utils.vlm_utils.preprocess_vlm_messages',return_value={}):
            record=evaluate_cases(api,p,dict(task_name='place_can_basket',render_freq=0),manifest,Path(tmp))[0]
        self.assertFalse(record['success'])
        self.assertIn('Insufficient coverage',record['failure_reason'])
        self.assertEqual(record['executed_commands'],0)
        self.assertEqual(env.total_commands,0)
        self.assertFalse(record['calls'][0]['decoded_valid'])
        report = record['calls'][0]['spline_diagnostics']
        self.assertFalse(report['coverage_valid'] or report['decoder_success'])
        self.assertIn('Insufficient coverage', report['exception'])

    def test_setup_failure_keeps_assigned_seed_and_the_other_episode(self):
        api=rollout_api()
        env=Environment()
        setup=env.setup_demo
        def fail_first(**kwargs):
            if kwargs['seed']==100000:
                raise RuntimeError('scene setup failed')
            setup(**kwargs)
        env.setup_demo=fail_first
        api.class_decorator=lambda name:env
        manifest=dict(cases=[dict(seed=100000,instruction='place'),dict(seed=100001,instruction='place')],instruction_type='seen')
        with tempfile.TemporaryDirectory() as tmp, patch('utils.vlm_utils.preprocess_vlm_messages',return_value={}):
            records=evaluate_cases(api,policy(),dict(task_name='place_can_basket',render_freq=0),manifest,Path(tmp))
        self.assertEqual([r['seed'] for r in records],[100000,100001])
        self.assertFalse(records[0]['success'])
        self.assertIn('scene setup failed',records[0]['failure_reason'])
        self.assertTrue(records[1]['success'])

    def test_nonfinite_actions_rejected(self):
        p=policy()
        p.model.prediction[0,0,0]=float('nan')
        p.begin_episode(100000,'place')
        p.call('update_obs',obs=rollout_api().robotwin_obs_to_xpolicylab(observation()))
        with patch('utils.vlm_utils.preprocess_vlm_messages',return_value={}):
            with self.assertRaisesRegex(ValueError,'Invalid decoded action'):
                p.call('get_action')

    def test_scene_selection_is_shared_and_not_replaced_after_policy_failure(self):
        env=Environment()
        def play():
            env.plan_success=env.seed!=100000
            return dict(info=dict(seed=env.seed))
        env.play_once=play
        env.check_success=lambda:env.plan_success
        api=SimpleNamespace(class_decorator=Mock(return_value=env),build_instruction=lambda args,info,kind,n:f"scene-{info['info']['seed']}")
        args=dict(task_name='place_can_basket',task_config='demo_clean',render_freq=0)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cases.json'
            a=evaluation_cases(api,args,path,episodes=2,seed=0,instruction_type='seen',max_attempts=4)
            b=evaluation_cases(api,args,path,episodes=2,seed=0,instruction_type='seen',max_attempts=4)
            self.assertEqual(a,b)
            self.assertEqual([c['seed'] for c in a['cases']],[100001,100002])
            api.class_decorator.assert_called_once()
            with self.assertRaisesRegex(ValueError,'do not match'):
                evaluation_cases(api,args,path,episodes=3,seed=0,instruction_type='seen',max_attempts=4)

    def test_loader_reconstructs_and_loads_the_requested_bundle(self):
        metadata=dict(config=dict(common={},model=dict(vlm=dict(checkpoint_path='local-vlm')),
                                  research=dict(contract={})),contract=dict(representation='dense'))
        model=Model(np.zeros((16,14)))
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)
            (checkpoint/'research_config.json').write_text(json.dumps(metadata))
            with patch('experiments.ac_spline_pilot.rollout_policy.encode_instructions',return_value={}) as encode, \
                 patch('experiments.ac_spline_pilot.trainer.create_ac_model',return_value=model) as build, \
                 patch('experiments.ac_spline_pilot.research.load_research_bundle') as reload, \
                 patch('transformers.AutoProcessor.from_pretrained',return_value=object()):
                result=load_policy(checkpoint,['place'])
            reload.assert_called_once_with(model,checkpoint)
            self.assertEqual(build.call_args.args[0].research.contract,metadata['contract'])
            self.assertFalse(result.model.training)
            encode.assert_called_once()


if __name__=='__main__': unittest.main()
