# Motus Policy for RoboTwin

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import sys
import os
import logging
import time
from typing import List, Dict, Any, Optional
from collections import deque
import yaml
from PIL import Image
from transformers import AutoProcessor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Use the same primary model implementation as training.
MOTUS_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MOTUS_ROOT))
# Add model paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.motus import Motus, MotusConfig

# Add bak path for T5EncoderModel
BAK_ROOT = str((Path(__file__).parent / "bak").resolve())
if BAK_ROOT not in sys.path:
    sys.path.insert(0, BAK_ROOT)

from wan.modules.t5 import T5EncoderModel
from data.utils.image_utils import resize_with_padding

logger = logging.getLogger(__name__)

class MotusPolicy:
    """
    Motus Policy wrapper for RoboTwin evaluation.
    Implements the joint video-action diffusion model for robotic control.
    """
    
    def __init__(self, checkpoint_path: str, config_path: str, wan_path: str, vlm_path: str, device: str = "cuda", log_dir: Optional[str] = None, task_name: Optional[str] = None):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.wan_path = wan_path
        self.vlm_path = vlm_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config_dict = yaml.safe_load(f)
        
        self.experiment_lora = self.config_dict.get('training', {}).get('lora', False)
        if self.experiment_lora:
            from utils.robotwin_prompt import CONDITIONING
            if self.config_dict.get('dataset', {}).get('conditioning') != CONDITIONING:
                raise ValueError('legacy raw-instruction checkpoint: incompatible conditioning; use its recorded source snapshot for diagnostics')
        self.spline = self.config_dict['common'].get('action_representation', 'dense') == 'spline'
        self.prediction_counts = {'invalid_predictions': 0, 'short_coverage': 0, 'repairs': 0, 'fallbacks': 0, 'knot_repairs': 0, 'gripper_saturations': 0}
        if self.experiment_lora:
            self.wan_path = self.config_dict['model']['wan']['config_path']
            self.vlm_path = self.config_dict['model']['vlm']['checkpoint_path']
        # Initialize model WITHOUT loading pretrained backbones
        self.model = self._load_model()

        # Initialize T5 encoder for language embeddings (WAN text encoder)
        self.t5_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device,
            checkpoint_path=os.path.join(self.wan_path, 'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=os.path.join(self.wan_path, 'google', 'umt5-xxl'),
        )

        # Initialize VLM processor from vlm_path (for tokenization only, weights from checkpoint)
        self.vlm_processor = AutoProcessor.from_pretrained(self.vlm_path, trust_remote_code=True)
        
        # Initialize observation cache
        self.obs_cache = deque(maxlen=1)
        self.action_cache = deque()
        
        # Model state
        self.current_state = None
        self.current_state_norm = None
        self.is_first_step = True
        self.prev_action = None

        # Load normalization stats
        if not self.experiment_lora:
            self._load_normalization_stats()
        if self.spline:
            import json
            from motus_actions.spline import AffineScale
            manifest = json.loads(Path(self.config_dict['dataset']['episode_manifest']).read_text())
            self.spline_parameter_scale = AffineScale(manifest['parameter_low'], manifest['parameter_high'])
            self.spline_action_scale = AffineScale(manifest['action_low'], manifest['action_high'])
        
        # Initialize image saving
        self.save_images = not self.experiment_lora
        base_log_dir = log_dir or os.environ.get('LOG_DIR') or str(Path(__file__).resolve().parent.parent / "logs")
        task_dir_name = task_name or os.environ.get('TASK_NAME') or "default_task"
        self.save_dir = Path(base_log_dir) / "images" / task_dir_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.episode_count = 0
        self.step_count = 0

        logger.info("Motus Policy initialized successfully")

    def set_instruction(self, instruction: str):
        """Set the current instruction for the policy."""
        self.current_instruction = instruction
        logger.info(f"Instruction set: {instruction}")

    def _load_model(self) -> Motus:
        """Load the Motus model without pretrained backbones, then load checkpoint."""
        logger.info(f"Initializing Motus model from config (no pretrained backbones)")

        config = self._create_model_config()
        
        # Initialize model from config WITHOUT loading pretrained weights
        model = Motus(config)
        model = model.to(self.device)
        
        if self.experiment_lora:
            from utils.lora import configure_lora
            from safetensors.torch import load_model
            if self.config_dict.get('evaluation', {}).get('checkpoint_kind') == 'pretrained':
                if self.spline:
                    raise ValueError('the original pretrained checkpoint requires dense actions')
                from utils.lora import load_initial_weights
                # Zero-output adapters keep the pretrained predictions unchanged
                # while retaining the shared experiment input/controller path.
                report = Path(self.config_dict['evaluation']['checkpoint_report'])
                load_initial_weights(model, self.checkpoint_path, report)
                configure_lora(model, rank=int(self.config_dict['training']['lora_rank']),
                               alpha=float(self.config_dict['training']['lora_alpha']))
                model.eval()
                return model
            configure_lora(model,rank=int(self.config_dict['training'].get('lora_rank',16)),
                           alpha=float(self.config_dict['training'].get('lora_alpha',32)))
            # Accelerator's native complete checkpoint includes adapters and projections.
            checkpoint = Path(self.checkpoint_path) / 'model.safetensors'
            load_model(model, str(checkpoint), strict=True, device=self.device)
            model.eval()
            return model
        if self.config_dict.get('evaluation', {}).get('checkpoint_kind') == 'pretrained':
            from utils.lora import load_initial_weights
            load_initial_weights(model, self.checkpoint_path, Path(self.config_dict['evaluation']['checkpoint_report']))
            model.eval()
            return model
        # Load checkpoint weights
        try:
            logger.info(f"Loading checkpoint from {self.checkpoint_path}")
            model.load_checkpoint(self.checkpoint_path, strict=False)
            logger.info("Model checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        model.eval()
        return model
    
    def _create_model_config(self) -> MotusConfig:
        """Create model configuration from yaml config - inference mode."""
        common = self.config_dict['common']
        model_cfg = self.config_dict['model']

        # Use paths passed to constructor
        vae_path = self.config_dict["model"]["wan"].get("vae_path", os.path.join(self.wan_path, "Wan2.2_VAE.pth"))
        vlm_checkpoint_path = self.vlm_path

        hidden_size = model_cfg['action_expert']['hidden_size']
        ffn_multiplier = model_cfg['action_expert']['ffn_dim_multiplier']

        config = MotusConfig(
            # Paths for config loading only (no weights loaded)
            wan_checkpoint_path=self.wan_path,
            vae_path=vae_path,
            wan_config_path=self.wan_path,
            video_precision='bfloat16',
            vlm_checkpoint_path=vlm_checkpoint_path,
            
            # Understanding expert config
            und_expert_hidden_size=512,
            und_expert_ffn_dim_multiplier=4,
            und_expert_norm_eps=1e-5,
            und_layers_to_extract=None,
            vlm_adapter_input_dim=2048,
            vlm_adapter_projector_type="mlp3x_silu",
            
            # Model architecture
            num_layers=30,
            action_state_dim=common['state_dim'],
            action_dim=common['action_dim'],
            action_representation=common.get('action_representation', 'dense'),
            action_expert_dim=hidden_size,
            action_expert_ffn_dim_multiplier=ffn_multiplier,
            action_expert_norm_eps=model_cfg["action_expert"]["norm_eps"] if self.experiment_lora else 1e-6,
            
            # Training config
            global_downsample_rate=common['global_downsample_rate'],
            video_action_freq_ratio=common['video_action_freq_ratio'],
            num_video_frames=common['num_video_frames'],
            video_loss_weight=1.0,
            action_loss_weight=1.0,
            
            # Inference config
            batch_size=1,
            video_height=common['video_height'],
            video_width=common['video_width'],
            
            # Don't load pretrained backbones - will load full model from checkpoint
            load_pretrained_backbones=False,
            training_mode='finetune',
        )

        return config
    
    def update_obs(self, observation: Dict[str, Any]):
        """Update observation cache with new observation."""
        # Extract visual observations
        if 'observation' in observation:
            obs_data = observation['observation']
            if 'head_camera' in obs_data and 'left_camera' in obs_data and 'right_camera' in obs_data:
                head_img = obs_data['head_camera']['rgb']
                left_img = obs_data['left_camera']['rgb']
                right_img = obs_data['right_camera']['rgb']
                
                from data.utils.multi_camera_concat import resize_and_concatenate_frames
                image = resize_and_concatenate_frames(None, head_img, left_img, right_img)
                if image is None:
                    raise ValueError('invalid camera mosaic')
            else:
                raise ValueError("Missing camera data")
        elif 'head_camera' in observation:
            image = observation['head_camera']
        elif 'image' in observation:
            image = observation['image']
        else:
            raise ValueError("No visual observation found")

        target_size = (self.config_dict['common']['video_height'],
                      self.config_dict['common']['video_width'])

        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        else:
            image_tensor = image

        if image_tensor.shape[-2:] != target_size:
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            resized_np = cv2.resize(image_np, (target_size[1], target_size[0])) if self.experiment_lora else resize_with_padding(image_np, target_size)
            if resized_np.dtype == np.uint8:
                resized_np = resized_np.astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized_np).permute(2, 0, 1).unsqueeze(0)
        
        if self.experiment_lora and image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.float() / 255.0
        self.obs_cache.append(image_tensor.to(self.device))

        # Extract robot state
        state = observation['joint_action']['vector']

        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state_tensor = state.float().unsqueeze(0) if state.dim() == 1 else state.float()

        self.current_state = state_tensor.to(self.device)
        self.current_state_norm = self.current_state if self.experiment_lora else self._normalize_actions(self.current_state).to(self.device)
    
    def get_action(self, instruction: str = None) -> List[np.ndarray]:
        """Get action predictions from the model."""
        if len(self.obs_cache) == 0:
            raise ValueError("No observations in cache. Call update_obs first.")
        
        if self.current_state is None:
            raise ValueError("No robot state available. Call update_obs first.")
        
        current_frame = self.obs_cache[-1]

        # Encode instruction with T5
        from utils.robotwin_prompt import robotwin_prompt
        instruction = robotwin_prompt(self.current_instruction if instruction is None else instruction)
        self.last_conditioning_text = instruction
        t5_out = self.t5_encoder([instruction], self.device)
        if isinstance(t5_out, torch.Tensor):
            t5_list = [t5_out.squeeze(0)] if t5_out.dim() == 3 else [t5_out]
        elif isinstance(t5_out, list):
            t5_list = t5_out
        else:
            raise ValueError("Unexpected T5 encoder output format")

        # Build VLM inputs
        first_frame_pil = self._tensor_to_pil_image(current_frame.squeeze(0).cpu())
        vlm_inputs = self._preprocess_vlm_messages(instruction, first_frame_pil)

        # Run inference
        num_inference_steps = self.config_dict['model']['inference']['num_inference_timesteps']
        with torch.no_grad():
            predicted_frames, predicted_actions = self.model.inference_step(
                first_frame=current_frame,
                state=self.current_state,
                num_inference_steps=num_inference_steps,
                language_embeddings=t5_list,
                vlm_inputs=[vlm_inputs],
            )

        # Save frame grid
        if predicted_frames is not None:
            if predicted_frames.dim() == 5:
                if predicted_frames.shape[1] == 3:
                    predicted_frames_viz = predicted_frames.permute(0, 2, 1, 3, 4)
                else:
                    predicted_frames_viz = predicted_frames
                
                condition_frame_viz = current_frame.squeeze(0)
                predicted_frames_viz = predicted_frames_viz.squeeze(0)
                
                self._save_frame_grid(condition_frame_viz, predicted_frames_viz)
                self.step_count += 1

        actions_real = predicted_actions.squeeze(0).cpu().numpy()
        self.last_decode_s = 0.
        if self.spline:
            decode_start = time.perf_counter()
            try:
                actions_real = self.decode_spline_prediction(actions_real)
            except ValueError as error:
                self.prediction_counts['invalid_predictions'] += 1
                if 'coverage' in str(error):
                    self.prediction_counts['short_coverage'] += 1
                raise
            self.last_decode_s = time.perf_counter()-decode_start
        if self.experiment_lora and (actions_real.shape != (16, 14) or not np.isfinite(actions_real).all()):
            self.prediction_counts['invalid_predictions'] += 1
            raise ValueError('invalid matched controller chunk')
        self.prev_action = actions_real[-1].copy()
        self.action_cache.extend(actions_real)

        return actions_real

    def decode_spline_prediction(self, packed, *, normalized=True, predicted=True):
        """The deployed decoder, also used for staged ground-truth diagnostics."""
        from motus_actions.spline import decode, repair_prediction_knots
        self.last_predicted_parameters = np.asarray(packed).copy()
        self.last_raw_parameters = None
        self.last_postprocessed_parameters = None
        parameters = self.spline_parameter_scale.decode(packed) if normalized else np.asarray(packed).copy()
        self.last_raw_parameters = parameters.copy()
        repairs = 0
        if predicted:
            parameters, repairs = repair_prediction_knots(parameters)
        self.last_postprocessed_parameters = parameters.copy()
        self.prediction_counts['knot_repairs'] += repairs
        self.prediction_counts['repairs'] += repairs
        offsets = np.arange(1, 17) * self.config_dict['common']['global_downsample_rate']
        return self.spline_action_scale.decode(decode(parameters, offsets))

    def _tensor_to_pil_image(self, tensor_chw: torch.Tensor) -> Image.Image:
        """Convert [C, H, W] tensor to PIL Image."""
        if tensor_chw.dtype != torch.float32:
            tensor_chw = tensor_chw.float()
        tensor_chw = tensor_chw.clamp(0, 1)
        np_img = (tensor_chw.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(np_img, mode='RGB')

    def _preprocess_vlm_messages(self, instruction: str, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Build VLM inputs."""
        if self.experiment_lora:
            from utils.vlm_utils import preprocess_vlm_messages
            encoded = preprocess_vlm_messages(instruction, image, self.vlm_processor)
            return {k: value.to(self.device) if torch.is_tensor(value) else value
                    for k, value in encoded.items()}
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': instruction},
                    {'type': 'image', 'image': image},
                ]
            }
        ]
        text = self.vlm_processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        encoded = self.vlm_processor(text=[text], images=[image], return_tensors='pt')
        vlm_inputs = {
            'input_ids': encoded['input_ids'].to(self.device),
            'attention_mask': encoded['attention_mask'].to(self.device), 
            'pixel_values': encoded['pixel_values'].to(self.device),
            'image_grid_thw': encoded.get('image_grid_thw', None)
        }
        if vlm_inputs['image_grid_thw'] is not None:
            vlm_inputs['image_grid_thw'] = vlm_inputs['image_grid_thw'].to(self.device)
        return vlm_inputs

    def _load_normalization_stats(self):
        """Load action normalization stats."""
        try:
            stat_path = Path(__file__).parent / 'utils' / 'stat.json'
            with open(stat_path, 'r') as f:
                stat_data = yaml.safe_load(f) if stat_path.suffix in ['.yml', '.yaml'] else None
        except Exception:
            stat_data = None
        if stat_data is None:
            import json as _json
            with open(Path(__file__).parent / 'utils' / 'stat.json', 'r') as f:
                stat_data = _json.load(f)

        stats = stat_data.get('robotwin2')
        if stats is None:
            raise ValueError('Normalization stats not found')
        self.action_min = torch.tensor(stats['min'], dtype=torch.float32, device=self.device)
        self.action_max = torch.tensor(stats['max'], dtype=torch.float32, device=self.device)
        self.action_range = self.action_max - self.action_min

    def _normalize_actions(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0,1]."""
        shape = x.shape
        x_flat = x.reshape(-1, shape[-1])
        norm = (x_flat - self.action_min.unsqueeze(0)) / self.action_range.unsqueeze(0)
        return norm.reshape(shape)

    def _denormalize_actions(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize from [0,1]."""
        shape = y.shape
        y_flat = y.reshape(-1, shape[-1])
        denorm = y_flat * self.action_range.unsqueeze(0) + self.action_min.unsqueeze(0)
        return denorm.reshape(shape)
    
    def _create_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor) -> Image.Image:
        """Create horizontal grid."""
        def tensor_to_numpy(tensor):
            if tensor.dim() == 3:
                tensor = tensor.permute(1, 2, 0)
            tensor = tensor.detach().cpu().float()
            tensor = torch.clamp(tensor, 0, 1)
            return (tensor.numpy() * 255).astype(np.uint8)
        
        condition_np = tensor_to_numpy(condition_frame)
        predicted_np = []
        num_pred_frames = predicted_frames.shape[0]
        for i in range(num_pred_frames):
            frame_np = tensor_to_numpy(predicted_frames[i])
            predicted_np.append(frame_np)
        
        while len(predicted_np) < 4:
            predicted_np.append(predicted_np[-1] if predicted_np else condition_np)
        
        all_frames = [condition_np] + predicted_np[:4]
        grid_image = np.concatenate(all_frames, axis=1)
        
        return Image.fromarray(grid_image)
    
    def _save_frame_grid(self, condition_frame: torch.Tensor, predicted_frames: torch.Tensor):
        """Save frame grid to disk."""
        if not self.save_images:
            return
        
        try:
            grid_image = self._create_frame_grid(condition_frame, predicted_frames)
            filename = f"episode_{self.episode_count:04d}_step_{self.step_count:04d}.png"
            save_path = self.save_dir / filename
            grid_image.save(save_path)
            logger.info(f"Saved frame grid to {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save frame grid: {e}")


def encode_obs(observation):
    """Post-Process Observation"""
    return observation


def get_model(usr_args):
    """
    Initialize Motus model.
    
    Args:
        usr_args: Arguments from eval script (must include wan_path and vlm_path)
    """
    checkpoint_path = usr_args.get('ckpt_setting')
    wan_path = usr_args.get('wan_path')  # Passed from eval.sh or auto_eval.sh
    vlm_path = usr_args.get('vlm_path')  # Passed from eval.sh or auto_eval.sh
    
    if not wan_path:
        raise ValueError("wan_path not provided in usr_args")
    
    if not vlm_path:
        raise ValueError("vlm_path not provided in usr_args")
    
    policy_dir = Path(__file__).parent
    config_path = Path(usr_args.get("config_path") or policy_dir / "utils" / "robotwin.yml")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    policy = MotusPolicy(
        checkpoint_path=checkpoint_path,
        wan_path=wan_path,
        vlm_path=vlm_path,
        config_path=str(config_path),
        device=device,
        log_dir=usr_args.get('log_dir'),
        task_name=usr_args.get('task_name')
    )
    
    return policy


def eval(TASK_ENV, model, observation):
    """Evaluation function."""
    if model.experiment_lora:
        execution=model.config_dict.get('execution',{})
        control_dt=execution.get('control_dt_s')
        if control_dt is None or not execution.get('clock_provenance'):
            raise ValueError('matched fixed-rate deployment requires a verified controller clock')
    if str(model.device).startswith('cuda'):
        torch.cuda.synchronize()
    policy_start = time.perf_counter()
    obs = encode_obs(observation)
    
    instruction = TASK_ENV.get_instruction()
    model.set_instruction(instruction)
    model.update_obs(obs)

    actions = model.get_action()
    if str(model.device).startswith('cuda'):
        torch.cuda.synchronize()
    if not hasattr(model, 'call_timings'):
        model.call_timings = []
    model.call_timings.append(dict(observation_to_executable_action_s=time.perf_counter()-policy_start,
        spline_decode_s=getattr(model, 'last_decode_s', 0.), warmup=len(model.call_timings)==0))
    if model.experiment_lora:
        from motus_actions.spline import controller_commands
        limits = [np.asarray(joint.get_limits())[0] for side in ('left', 'right')
                  for joint in getattr(TASK_ENV.robot, f'{side}_arm_joints')]
        try:
            actions, repairs = controller_commands(actions, limits)
        except ValueError:
            model.prediction_counts['invalid_predictions'] += 1
            raise
        model.prediction_counts['repairs'] += repairs
        model.prediction_counts['gripper_saturations'] += repairs
    
    if model.experiment_lora:
        from types import SimpleNamespace
        reference=str(MOTUS_ROOT.parent/'WAM-RoboTwin')
        if reference not in sys.path: sys.path.insert(0,reference)
        from trajectory_tools.replay import execute_fixed_rate
        if TASK_ENV.take_action_cnt+len(actions)>TASK_ENV.step_lim:
            raise TimeoutError('insufficient controller-step budget for a matched prefix')
        plan=SimpleNamespace(actions=actions,control_dt=float(control_dt),
            execution_times=np.arange(len(actions))*float(control_dt),requested_duration=len(actions)*float(control_dt))
        result=execute_fixed_rate(TASK_ENV,plan)
        model.last_controller_result=result
        TASK_ENV.take_action_cnt+=result['action_commands']
        TASK_ENV.eval_success=bool(result['success'])
        if result['status']!='completed': raise ValueError(result['error'])
    else:
        before = TASK_ENV.take_action_cnt
        for action in actions:
            TASK_ENV.take_action(action, action_type='qpos')
        model.last_controller_result = dict(action_commands=TASK_ENV.take_action_cnt-before,
            simulated_duration_s=None, controller_protocol='upstream_take_action_TOPP')


def reset_model(model):  
    """Reset model cache at episode start."""
    model.obs_cache.clear()
    model.action_cache.clear()
    model.current_state = None
    model.is_first_step = True
    model.prev_action = None
    model.episode_count += 1
    model.step_count = 0
    logger.info(f"Model reset completed for episode {model.episode_count}")
