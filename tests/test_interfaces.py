"""Tests the real dataset, deployment call, and Euler loop without huge weights."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from collections import deque

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]


def oracle_inference(clean):
    """Run Motus.inference_step itself with analytically correct velocity heads."""
    from models.motus import Motus
    model=Motus.__new__(Motus)
    nn.Module.__init__(model)
    model.config=SimpleNamespace(num_video_frames=8,action_chunk_size=36,action_token_dim=15,num_layers=0)
    model.device=torch.device('cpu')
    model.dtype=torch.float32
    model.video_model=SimpleNamespace(precision=torch.float32,
        encode_video=lambda image: torch.zeros(1,1,1,1,1),decode_video=lambda x:x)
    model.video_module=SimpleNamespace(prepare_input=lambda x:x,
        preprocess_t5_embeddings=lambda x:x, get_time_embedding=lambda t,n:(t,None),
        apply_output_head=lambda x,t:torch.zeros_like(x))
    model.und_module=SimpleNamespace(extract_und_features=lambda x:None)
    model.action_module=SimpleNamespace(get_time_embedding=lambda t,n:(t,None))
    def encode(state,actions,registers):
        return torch.cat([torch.zeros(1,1,15),actions,registers],dim=1)
    def velocity(tokens,time):
        # E[noise-clean | x_t, clean]=(x_t-clean)/sigma, the exact oracle.
        result=torch.zeros_like(tokens)
        result[:,1:-4]=(tokens[:,1:-4]-clean)/(time.reshape(-1,1,1)/1000)
        return result
    model.action_expert=SimpleNamespace(config=SimpleNamespace(num_registers=4),
        registers=torch.zeros(1,4,15),input_encoder=encode,decoder=velocity)
    with torch.no_grad():
        _,prediction=model.inference_step(torch.zeros(1,3,4,4),torch.zeros(1,14),
            num_inference_steps=10,language_embeddings=[torch.zeros(1,2)],vlm_inputs=None)
    return prediction


class InterfaceTests(unittest.TestCase):
    def test_exact_inference_euler_oracle(self):
        clean=torch.linspace(-1,1,36*15).reshape(1,36,15)
        torch.manual_seed(17)
        torch.testing.assert_close(oracle_inference(clean),clean,atol=5e-7,rtol=0)

    def test_boundary_and_interior_multiplicity(self):
        from motus_actions.spline import local_curve
        knots=np.r_[np.zeros(4), np.arange(1,29), np.full(4,29.)]
        local=np.column_stack([knots,np.zeros((36,14))])
        local_curve(local)
        local[8:11,0]=local[8,0]  # allowed cubic interior multiplicity 3
        local_curve(local)
        local[11,0]=local[8,0]
        with self.assertRaisesRegex(ValueError,'multiplicity'):
            local_curve(local)
        # Surplus endpoint padding is supported without discarding interior CPs.
        local[:,0]=np.r_[np.zeros(4),np.arange(1,25),np.full(8,25.)]
        curve=local_curve(local)
        self.assertEqual(len(curve.t),32)
        self.assertEqual(len(curve.c),28)

    def test_actual_dataset_to_deployment_conditioning(self):
        import h5py
        from omegaconf import OmegaConf
        from data.dataset import create_dataset
        from inference.robotwin.Motus.deploy_policy import MotusPolicy
        from motus_actions.prepare import digest
        from motus_actions.spline import AffineScale
        from utils.robotwin_prompt import robotwin_prompt, CONDITIONING
        from bak.wan.modules.t5 import T5EncoderModel
        from bak.wan.modules.tokenizers import HuggingfaceTokenizer
        import sys
        sys.path.insert(0,str(ROOT.parent/'WAM-RoboTwin'))
        from XPolicyLab.utils.process_data import decode_image_bit
        source=ROOT/'outputs/pilot-cache-guard-20261111/manifest.json'
        self.assertTrue(source.exists(),'requires preserved audited real data cache')
        manifest=json.loads(source.read_text())
        tokenizer=HuggingfaceTokenizer(str(ROOT/'.runtime/model-metadata/Wan2.2-TI2V-5B/google/umt5-xxl'),
            seq_len=512,clean='whitespace')
        class TokenSpy(nn.Module):
            def forward(self,ids,mask):
                self.ids,self.mask=ids.clone(),mask.clone()
                return ids.float().unsqueeze(-1).repeat(1,1,2)
        t5=T5EncoderModel.__new__(T5EncoderModel)
        t5.tokenizer=tokenizer
        t5.model=TokenSpy()
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            (directory/'language').mkdir()
            for instruction in {e['instruction'] for e in manifest['episodes']}:
                prompt=robotwin_prompt(instruction)
                embedding=t5([prompt],'cpu')[0]
                torch.save(embedding,directory/'language'/f'{hashlib.sha256(prompt.encode()).hexdigest()}.pt')
            manifest['converter_sha256']=digest(ROOT/'motus_actions/prepare.py')
            manifest['spline_sha256']=digest(ROOT/'motus_actions/spline.py')
            manifest['conditioning']=CONDITIONING
            manifest.pop('identity')
            manifest['identity']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
            path=directory/'manifest.json'
            path.write_text(json.dumps(manifest))
            for representation in ('dense','spline'):
                config=OmegaConf.load(ROOT/'configs/robotwin_ac.yaml')
                config.dataset.episode_manifest=str(path)
                config.common.action_representation=representation
                dataset=create_dataset(config)
                self.assertEqual(len(dataset),450)
                random.seed(13)
                sample=dataset[0]  # exercise public dataset entry point
                entry=next(e for e in manifest['episodes'] if e['id']==sample['episode_id'])
                current=sample['observation_index']
                with h5py.File(entry['hdf5_path'],'r') as f:
                    cameras={key:{'rgb':decode_image_bit(f[f'vision/{camera}/colors'][current])}
                        for key,camera in [('head_camera','cam_head'),('left_camera','cam_left_wrist'),('right_camera','cam_right_wrist')]}
                policy=MotusPolicy.__new__(MotusPolicy)
                policy.config_dict=OmegaConf.to_container(config,resolve=True)
                policy.experiment_lora=True
                policy.spline=representation=='spline'
                policy.obs_cache,policy.action_cache=deque(maxlen=1),deque()
                policy.device='cpu'
                policy.vlm_processor=dataset.vlm_processor
                policy.t5_encoder=t5
                policy.prediction_counts=dict(knot_repairs=0,repairs=0,invalid_predictions=0,short_coverage=0)
                policy.spline_parameter_scale=AffineScale(manifest['parameter_low'],manifest['parameter_high'])
                policy.spline_action_scale=AffineScale(manifest['action_low'],manifest['action_high'])
                class ModelSpy:
                    def inference_step(self,**inputs):
                        self.inputs=inputs
                        return None,sample['action_sequence'][None]
                policy.model=ModelSpy()
                policy.set_instruction(entry['instruction'])
                policy.update_obs({'observation':cameras,'joint_action':{'vector':sample['initial_state'].numpy()}})
                policy.get_action()  # exercise public deployment entry point
                self.assertEqual(policy.last_conditioning_text,sample['text_instruction'])
                self.assertEqual(robotwin_prompt(sample['text_instruction']),sample['text_instruction'])
                inputs=policy.model.inputs
                torch.testing.assert_close(inputs['first_frame'][0],sample['first_frame'],rtol=0,atol=0)
                torch.testing.assert_close(inputs['state'][0],sample['initial_state'],rtol=0,atol=0)
                torch.testing.assert_close(policy.current_state_norm[0],sample['initial_state'],rtol=0,atol=0)
                expected_ids,expected_mask=tokenizer([sample['text_instruction']],return_mask=True,add_special_tokens=True)
                torch.testing.assert_close(t5.model.ids,expected_ids,rtol=0,atol=0)
                torch.testing.assert_close(t5.model.mask,expected_mask,rtol=0,atol=0)
                torch.testing.assert_close(inputs['language_embeddings'][0],sample['language_embedding'],rtol=0,atol=0)
                for key,value in sample['vlm_inputs'].items():
                    torch.testing.assert_close(inputs['vlm_inputs'][0][key],value,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
