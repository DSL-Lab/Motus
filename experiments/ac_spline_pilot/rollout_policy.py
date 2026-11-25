"""A/C research bundles exposed through RoboTwin's existing policy-call API."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from data.robotwin2.adaptive_spline import decode_execution

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
    model, config, language, metadata = load_checkpoint(checkpoint, instructions)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config.model.vlm.checkpoint_path, trust_remote_code=True)
    return RolloutPolicy(model, config, metadata['contract'], language, processor)


def load_checkpoint(checkpoint, instructions=()):
    from omegaconf import OmegaConf
    from .trainer import create_ac_model
    from .research import load_research_bundle
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint / 'research_config.json').read_text())
    config = OmegaConf.create(metadata['config'])
    config.research.contract = metadata['contract']
    language = encode_instructions(instructions, config) if instructions else {}
    model = create_ac_model(config)
    load_research_bundle(model, checkpoint)
    model.eval()
    return model, config, language, metadata


class RolloutPolicy:
    """In-process client; observations and actions use the existing evaluator."""
    _robotwin_protocol = 'inprocess'

    def __init__(self, model, config, contract, language, processor):
        self.model, self.config, self.contract = model, config, contract
        self.language, self.processor = language, processor
        self.device = next(model.parameters()).device
        self.seed = 0
        self.episode_id = None
        self.spline_diagnostics = bool(getattr(getattr(config, 'research', None), 'spline_diagnostics', False))
        self.execution_prefix = contract.get('executed_prefix')
        self.reset()

    def begin_episode(self, seed, instruction, episode_id=None):
        self.seed, self.instruction = int(seed), instruction
        self.episode_id = self.seed if episode_id is None else episode_id
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
        call = dict(index=len(self.calls), decoded_valid=False)
        self.calls.append(call)
        started = time.perf_counter()
        try:
            return self._get_action(call)
        except Exception as exc:
            call['error'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            call['total_wall_seconds'] = time.perf_counter() - started

    def _get_action(self, call):
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
        spline = self.contract['representation'] == 'spline'
        diagnostics = None
        if spline and self.spline_diagnostics:
            diagnostics = dict(episode_id=self.episode_id, policy_call_index=call['index'],
                               coverage_valid=None, decoder_success=False)
            call['spline_diagnostics'] = diagnostics
        devices = [self.device.index or 0] if self.device.type == 'cuda' else []
        frame = frame[None].to(self.device)
        state = torch.from_numpy(state.astype(np.float32))[None].to(self.device)
        language = [self.language[self.instruction].to(self.device)]
        try:
            # Keep model noise matched by scene/call, without changing simulator RNG.
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed((self.seed << 32) + call['index'])
                if devices:
                    torch.cuda.synchronize(self.device)
                started = time.perf_counter()
                _, prediction = self.model.inference_step(
                    first_frame=frame, state=state,
                    num_inference_steps=int(self.config.model.inference.num_inference_timesteps),
                    language_embeddings=language, vlm_inputs=vlm)
                if devices:
                    torch.cuda.synchronize(self.device)
                call['inference_wall_seconds'] = time.perf_counter() - started
            parameters = prediction[0].float().cpu().numpy()
            if spline:
                started = time.perf_counter()
                try:
                    commands, report = decode_execution(parameters, self.contract['knot_scaling'],
                        self.contract['normalization'], self.execution_prefix, diagnostics=diagnostics)
                finally:
                    call['decode_wall_seconds'] = time.perf_counter() - started
                call['decoder_report'] = report
            else:
                commands = parameters
            count = len(commands) if spline and 'row_budget' in self.execution_prefix else self.execution_prefix['command_count']
            if not count or commands.shape != (count, 14) or not np.isfinite(commands).all():
                raise ValueError('Invalid decoded action chunk')
            call['decoded_valid'] = True
            if diagnostics is not None:
                diagnostics['decoder_success'] = True
            return commands
        except Exception as exc:
            if diagnostics is not None:
                diagnostics['exception'] = str(exc)
            raise
