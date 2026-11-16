"""A/C research bundles exposed through RoboTwin's existing policy-call API."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from data.robotwin2.adaptive_spline import decode_local, prepare_local

ROBOTWIN_ROOT = Path(__file__).resolve().parents[3] / 'WAM-RoboTwin'


def preparation_helpers():
    # Reuse the exact training camera composition and prompt prefix.
    path = str(ROBOTWIN_ROOT / 'scripts')
    if path not in sys.path:
        sys.path.insert(0, path)
    from prepare_motus_clean50 import PREFIX, compose_rgb
    return PREFIX, compose_rgb


def encode_instructions(instructions, config):
    """Encode before loading MOTUS so the large T5 model is freed first."""
    bak = str(Path(__file__).resolve().parents[2] / 'bak')
    if bak not in sys.path:
        sys.path.insert(0, bak)
    from wan.modules.t5 import T5EncoderModel
    prefix, _ = preparation_helpers()
    assets = Path(config.model.wan.vae_path).parent
    encoder = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=torch.device('cuda:0'),
        checkpoint_path=str(assets / 'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=str(assets / 'google/umt5-xxl'))
    embeddings = {}
    with torch.inference_mode():
        for instruction in dict.fromkeys(instructions):
            embeddings[instruction] = encoder([prefix + instruction], torch.device('cuda:0'))[0].detach().cpu()
    del encoder
    torch.cuda.empty_cache()
    return embeddings


def load_policy(checkpoint, instructions):
    """Rebuild the saved architecture/base/adapters; validate the exact bundle."""
    from omegaconf import OmegaConf
    from transformers import AutoProcessor
    from .trainer import create_ac_model
    from .research import load_research_bundle
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / 'research_config.json').read_text())
    config = OmegaConf.create(metadata['config'])
    config.research.contract = metadata['contract']
    language = encode_instructions(instructions, config)
    model = create_ac_model(config)
    load_research_bundle(model, checkpoint)
    model.eval()
    processor = AutoProcessor.from_pretrained(config.model.vlm.checkpoint_path, trust_remote_code=True)
    return RolloutPolicy(model, config, metadata['contract'], language, processor)


class RolloutPolicy:
    """In-process client; observations and actions use the existing evaluator."""
    _robotwin_protocol = 'inprocess'

    def __init__(self, model, config, contract, language, processor):
        self.model, self.config, self.contract = model, config, contract
        self.language, self.processor = language, processor
        self.device = next(model.parameters()).device
        self.seed = 0
        self.reset()

    def begin_episode(self, seed, instruction):
        self.seed, self.instruction = int(seed), instruction
        self.reset()

    def reset(self):
        self.observation = None
        self.calls = []

    def call(self, func_name, **kwargs):
        if func_name == 'reset':
            self.reset()
        elif func_name == 'update_obs':
            self.observation = kwargs['obs']
        elif func_name == 'get_action':
            return self.get_action()
        else:
            raise NotImplementedError(func_name)

    def get_action(self):
        from data.utils.image_utils import resize_with_padding, tensor_to_pil
        from utils.vlm_utils import preprocess_vlm_messages
        if self.observation is None:
            raise ValueError('Policy needs the current RoboTwin observation')
        prefix, compose = preparation_helpers()
        obs = self.observation
        images = [np.asarray(obs['vision'][camera]['color']) for camera in
                  ('cam_head', 'cam_left_wrist', 'cam_right_wrist')]
        if any(image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8 for image in images):
            raise ValueError('Expected decoded uint8 RGB cameras')
        image = compose(*images)
        size = (int(self.config.common.video_height), int(self.config.common.video_width))
        if image.shape[:2] != size:
            image = resize_with_padding(image, size)
        frame = torch.from_numpy(image).permute(2, 0, 1).float() / 255.
        state = np.concatenate([np.asarray(obs['state'][key]).reshape(-1) for key in
            ('left_arm_joint_state', 'left_ee_joint_state', 'right_arm_joint_state', 'right_ee_joint_state')])
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError('Expected finite raw 6+1+6+1 robot state')
        vlm = preprocess_vlm_messages(prefix + self.instruction, tensor_to_pil(frame), self.processor)
        vlm = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in vlm.items()}
        call = dict(index=len(self.calls), decoded_valid=False)
        self.calls.append(call)
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        if devices:
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            # Keep model noise matched by scene/call, without changing simulator RNG.
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed((self.seed << 32) + call['index'])
                _, prediction = self.model.inference_step(
                    first_frame=frame[None].to(self.device),
                    state=torch.from_numpy(state.astype(np.float32))[None].to(self.device),
                    num_inference_steps=int(self.config.model.inference.num_inference_timesteps),
                    language_embeddings=[self.language[self.instruction].to(self.device)], vlm_inputs=vlm)
            if devices:
                torch.cuda.synchronize(self.device)
            parameters = prediction[0].float().cpu().numpy()
            call['inference_wall_seconds'] = time.perf_counter() - started
            started = time.perf_counter()
            count = self.contract['executed_prefix']['command_count']
            if self.contract['representation'] == 'spline':
                _, _, report = prepare_local(parameters, self.contract['knot_scaling'])
                call['decoder_report'] = report
                times = self.contract['executed_prefix']['source_stride'] * np.arange(1, count + 1)
                commands, report = decode_local(parameters, times, 0., self.contract['knot_scaling'], self.contract['normalization'])
                call['decoder_report'] = report
            else:
                commands = parameters
            if commands.shape != (count, 14) or not np.isfinite(commands).all():
                raise ValueError('Invalid decoded action chunk')
            call['decoded_valid'] = True
            return commands
        except Exception as exc:
            call['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            call['decode_wall_seconds' if 'inference_wall_seconds' in call else 'failed_inference_wall_seconds'] = time.perf_counter() - started
