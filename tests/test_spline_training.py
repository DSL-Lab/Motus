"""CPU checks using the real Motus losses, experts, and joint attention.

The miniature fixture replaces pretrained VAE/VLM features with small fixed
encoders. It does not claim to validate checkpoint-scale training or robotics.
Run: python -m unittest discover -s tests -p test_spline_training.py -v
"""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import BSpline
import torch
from torch import nn

from motus_actions.lora import check_gradients, configure_lora
from motus_actions.spline_model import adapt_action_representation, load_adapter, save_adapter
from motus_actions.spline_window import (decode_window, fit_adaptive_spline, local_window,
                                         normalize_actions, repair_knots, select_video_frames)
from motus_actions.temporal import latent_video_validity, masked_video_loss


def metadata(rows=24):
    return dict(rows=rows, num_video_frames=8, action_min=[-1., -1.], action_max=[1., 1.],
                time_scale=2., controller_hz=30., max_duration=10., gripper_indices=[1])


class SplineChecks(unittest.TestCase):
    def test_adaptive_reconstruction_and_local_slice(self):
        times = np.arange(100) / 30
        actions = np.column_stack((np.sin(2 * times), np.where(times < 1.7, -1., 1.)))
        fitted = fit_adaptive_spline(times, actions, .02)
        self.assertLessEqual(np.max(np.abs(fitted(times) - actions)), .0200001)
        self.assertTrue(np.all(np.diff(fitted.t) >= 0))
        for observation in (0., times[24], times[-2]):
            window = local_window(fitted.t, fitted.c, observation, 24, 1 / 30)
            ticks = np.linspace(0, min(window[-4, 0], times[-1] - observation) - 1e-9, 7)
            decoded = BSpline(window[:, 0], window[:-4, 1:], 3)(ticks)
            np.testing.assert_allclose(decoded, fitted(ticks + observation), atol=1e-7)
            # Constant hold beyond the real episode end, without invalid padding.
            if window[-4, 0] > times[-1] - observation:
                np.testing.assert_allclose(BSpline(window[:, 0], window[:-4, 1:], 3)(
                    [times[-1] - observation + .001]), fitted(times[-1])[None], atol=1e-6)

    def test_short_constant_and_normalization(self):
        fitted = fit_adaptive_spline([0, .1], np.ones((2, 2)), .01)
        np.testing.assert_allclose(fitted([0, .1]), 1)
        np.testing.assert_allclose(normalize_actions(np.array([[2, 5]]), np.array([2, 0]), np.array([2, 10])), [[0, 0]])
        with self.assertRaises(ValueError):
            fit_adaptive_spline([0, 0], np.ones((2, 2)))

    def test_frame_rounding_boundaries_and_padding(self):
        window = np.zeros((24, 3))
        window[[5, 7, 9, 11, 13, 15, 17, 19], 0] = [.049, .05, .151, .3, .4, .5, .6, .7]
        indices, valid = select_video_frames(window, 0, np.arange(5) / 10, 8, 2)
        np.testing.assert_array_equal(indices, [0, 0, 0, 2, 3, 4, 4, 4, 4])
        np.testing.assert_array_equal(valid, [1, 0, 0, 1, 1, 1, 0, 0, 0])

    def test_compressed_padding_and_masked_loss(self):
        raw = torch.ones(2, 9, dtype=torch.bool)
        raw[0, 7:] = False
        raw[1, 2] = False
        latent = latent_video_validity(raw)
        torch.testing.assert_close(latent, torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool))
        target = torch.zeros(2, 2, 3, 1, 1)
        prediction = torch.ones_like(target, requires_grad=True)
        loss = masked_video_loss(prediction, target, latent)
        loss.backward()
        self.assertEqual(loss.item(), 1)
        self.assertEqual(prediction.grad[:, :, 0].abs().sum(), 0)
        self.assertEqual(prediction.grad[0, :, 2].abs().sum(), 0)
        self.assertEqual(prediction.grad[1].abs().sum(), 0)

    def test_ordering_execution_budget_and_gripper(self):
        values = np.array([3, -2, 1, 1, -9, 0, -1, 0, 0, 0, -8, 2], float)
        knots = repair_knots(values, 30, 10)
        self.assertTrue(np.all(np.diff(knots) >= 0))
        self.assertLessEqual(knots[3], 0)
        self.assertGreater(knots[-4], 0)
        data = metadata(12)
        window = np.column_stack((np.linspace(-.3, .8, 12), np.ones((12, 2)) * .2))
        commands, ticks, next_call = decode_window(window, data, 12)
        self.assertTrue(np.isfinite(commands).all())
        np.testing.assert_allclose(commands[:, 1], 1)
        self.assertAlmostEqual(next_call, len(commands) / 30)
        self.assertTrue(np.all(np.diff(ticks) > 0))
        longer = window.copy()
        longer[:, 0] *= 2
        self.assertGreater(len(decode_window(longer, data, 12)[0]), len(commands))
        window[0, 0] = np.nan
        with self.assertRaises(ValueError):
            decode_window(window, data, 12)


def tiny_joint_model(video_lora=False):
    from models.action_expert import ActionExpert, ActionExpertConfig
    from models.motus import Motus, MotusConfig, VideoModule, ActionModule, UndModule
    from models.und_expert import UndExpert, UndExpertConfig
    from wan.modules.model import WanModel
    from wan.utils.fm import FlowMatchScheduler

    class TinyVideo(nn.Module):
        def __init__(self):
            super().__init__()
            self.precision = torch.float32
            self.wan_model = WanModel(dim=32, ffn_dim=64, num_heads=4, num_layers=2,
                                      patch_size=(1, 1, 1), in_dim=3, out_dim=3, text_dim=16)
            nn.init.normal_(self.wan_model.head.head.weight, std=.05)

        def encode_video(self, pixels):
            # Only the external VAE is replaced: one current + groups of four.
            count = (pixels.shape[2] - 1) // 4
            if not count:
                return pixels[:, :, :1]
            future = pixels[:, :, 1:].unflatten(2, (count, 4)).mean(dim=3)
            return torch.cat((pixels[:, :, :1], future), dim=2)

        def decode_video(self, latent):
            raise AssertionError("RGB decoding must remain optional")

    torch.manual_seed(7)
    model = Motus.__new__(Motus)
    nn.Module.__init__(model)
    model.config = MotusConfig(device="cpu", num_layers=2, action_state_dim=2, action_dim=2,
                                action_expert_dim=32, num_video_frames=8, batch_size=2)
    model.device, model.dtype = torch.device("cpu"), torch.float32
    model.video_model = TinyVideo()
    wan = dict(dim=32, num_heads=4, head_dim=8)
    model.action_expert = ActionExpert(ActionExpertConfig(dim=32, ffn_dim=64, num_layers=2,
                                                         state_dim=2, action_dim=2, chunk_size=17), wan)
    nn.init.normal_(model.action_expert.decoder.action_head[-1].weight, std=.05)
    model.und_expert = UndExpert(UndExpertConfig(dim=16, ffn_dim=32, num_layers=2, vlm_input_dim=16), wan)
    model.vlm_model = nn.Identity()
    model.video_module = VideoModule(model.video_model, model.dtype, model.device, None)
    model.action_module = ActionModule(model.action_expert, model.config, model.video_model,
                                       model.vlm_model, model.dtype, model.device)
    model.und_module = UndModule(model.vlm_model, model.und_expert, model.config, model.dtype, model.device)
    model.und_module.extract_und_features = lambda inputs: torch.zeros(inputs.shape[0], 2, 16)
    for name in ("fm_train_scheduler", "fm_train_scheduler_action"):
        scheduler = FlowMatchScheduler(num_train_timesteps=10)
        scheduler.set_timesteps(10, training=True)
        setattr(model, name, scheduler)
    adapt_action_representation(model, metadata())
    configure_lora(model, rank=2, video_lora=video_lora)
    return model


def tiny_batch():
    torch.manual_seed(4)
    return dict(first_frame=torch.rand(2, 3, 2, 2), video_frames=torch.rand(2, 8, 3, 2, 2),
                state=torch.rand(2, 2), actions=torch.randn(2, 24, 3) * .3,
                language_embeddings=torch.randn(2, 3, 16), vlm_inputs=torch.zeros(2, 1),
                video_frame_valid=torch.ones(2, 9, dtype=torch.bool))


class JointTrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_backward_video_gradient_and_frozen_weights(self):
        for video_lora in (False, True):
            model = tiny_joint_model(video_lora)
            losses = model.training_step(**tiny_batch())
            adapter = model.action_expert.blocks[0].wan_action_lora.lora_b
            gradient = torch.autograd.grad(losses["video_loss"], adapter, retain_graph=True)[0]
            self.assertGreater(gradient.abs().sum().item(), 0)
            losses["total_loss"].backward()
            check_gradients(model)
            self.assertTrue(all(not value.requires_grad for value in model.vlm_model.parameters()))
            # Adapter serialization restores all trainable components exactly.
            with tempfile.TemporaryDirectory() as directory:
                file = Path(directory) / "adapter.pt"
                save_adapter(file, model, rank=2, video_lora=video_lora, step=1)
                saved = torch.load(file, weights_only=True)
                load_adapter(model, saved)

    def test_padded_keys_cannot_influence_other_tokens(self):
        from wan.modules.model import WanSelfAttention, rope_params
        attention = WanSelfAttention(32, 4)
        value = torch.randn(1, 3, 32)
        action = torch.randn(1, 2, 4, 8)
        grid = torch.tensor([[3, 1, 1]])
        frequencies = torch.cat([rope_params(1024, 4), rope_params(1024, 2), rope_params(1024, 2)], dim=1)
        kwargs = dict(seq_lens=torch.tensor([5]), grid_sizes=grid, freqs=frequencies,
                      action_q=action, action_k=action, action_v=action,
                      video_key_valid=torch.tensor([[True, True, False]]))
        first = attention(value, **kwargs)
        value[:, 2] += 100
        second = attention(value, **kwargs)
        torch.testing.assert_close(first[1], second[1])
        torch.testing.assert_close(first[0][:, :2], second[0][:, :2])

    def test_fixed_tiny_overfit_and_joint_inference_without_rgb(self):
        model, batch = tiny_joint_model(), tiny_batch()
        frozen = {name: value.clone() for name, value in model.named_parameters() if not value.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        history = []
        for _ in range(80):
            torch.manual_seed(11)
            optimizer.zero_grad(set_to_none=True)
            losses = model.training_step(**batch)
            losses["total_loss"].backward()
            optimizer.step()
            history.append(float(losses["total_loss"].detach()))
        self.assertLess(history[-1], history[0] * .55)
        for name, value in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)
        print(f"Synthetic tiny joint overfit (80 steps): {history[0]:.6f} -> {history[-1]:.6f}")
        with torch.inference_mode():
            frames, parameters = model.inference_step(
                batch["first_frame"], state=batch["state"], num_inference_steps=2,
                language_embeddings=batch["language_embeddings"], vlm_inputs=batch["vlm_inputs"], decode_rgb=False)
        self.assertIsNone(frames)
        self.assertEqual(parameters.shape, (2, 24, 3))
        self.assertTrue(torch.isfinite(parameters).all())


if __name__ == "__main__":
    unittest.main()
