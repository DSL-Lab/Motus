"""Actual wrapper methods with model/OpenCV stubs; no GPU or interpolation claim."""
import ast
from collections import deque
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import Mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'inference/robotwin/Motus/deploy_policy.py'


def resize_fixture(frame, size):
    # Constant-color fixtures isolate geometry/range from unavailable OpenCV.
    return np.broadcast_to(frame[0, 0], (size[1], size[0], 3)).copy()


def wrapper_methods():
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MotusPolicy')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in {'update_obs', '_load_model'}]
    image_source = ROOT / 'inference/robotwin/Motus/utils/image_utils.py'
    resize = next(n for n in ast.parse(image_source.read_text(encoding='utf-8')).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'resize_with_padding')
    namespace = dict(torch=torch, np=np, cv2=SimpleNamespace(resize=resize_fixture), deque=deque, Dict=dict, Any=object,
                     Tuple=tuple, Motus=object, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[resize] + methods, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class DeploymentContract(TestCase):
    def setUp(self):
        self.ns = wrapper_methods()
        self.policy = Mock()
        self.policy.config_dict = {'common': {'video_height': 384, 'video_width': 320}}
        self.policy.device = 'cpu'
        self.policy.obs_cache = deque(maxlen=1)
        self.policy._normalize_actions = lambda x: x

    def observe(self, image):
        self.ns['update_obs'](self.policy, dict(image=image, joint_action={'vector': np.arange(14, dtype=np.float32)}))
        return self.policy.obs_cache[-1]

    def test_already_sized_uint8_matches_training_range(self):
        result = self.observe(np.full((384, 320, 3), 128, dtype=np.uint8))
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, torch.full_like(result, 128 / 255))
        torch.testing.assert_close(self.policy.current_state, torch.arange(14).float()[None])

    def test_resized_uint8_uses_padding_and_same_range(self):
        image = np.full((150, 200, 3), 128, dtype=np.uint8)
        expected = self.ns['resize_with_padding'](image, (384, 320))
        expected = torch.from_numpy(expected).permute(2, 0, 1)[None].float() / 255
        torch.testing.assert_close(self.observe(image), expected)

    def test_float_rgb_preserved_and_bad_range_rejected(self):
        image = np.full((384, 320, 3), .25, dtype=np.float32)
        torch.testing.assert_close(self.observe(image), torch.full((1, 3, 384, 320), .25))
        for value in (2., float('nan')):
            with self.assertRaises(ValueError):
                self.observe(np.full_like(image, value))

    def test_three_view_composition_uses_actual_head_dimensions(self):
        self.policy.config_dict['common'] = {'video_height': 720, 'video_width': 640}
        cameras = {name: {'rgb': np.full((480, 640, 3), value, dtype=np.uint8)}
                   for name, value in [('head_camera', 255), ('left_camera', 64), ('right_camera', 128)]}
        self.ns['update_obs'](self.policy, dict(observation=cameras, joint_action={'vector': np.zeros(14)}))
        image = self.policy.obs_cache[-1]
        self.assertEqual(tuple(image.shape), (1, 3, 720, 640))
        self.assertAlmostEqual(float(image[0, 0, 0, 0]), 1)
        self.assertAlmostEqual(float(image[0, 0, 500, 0]), 64 / 255)
        self.assertAlmostEqual(float(image[0, 0, 500, 500]), 128 / 255)

    def test_original_checkpoint_loading_is_strict(self):
        model = Mock()
        model.to.return_value = model
        self.ns['Motus'] = Mock(return_value=model)
        self.policy.checkpoint_path = 'test-only-checkpoint'
        self.ns['_load_model'](self.policy)
        model.load_checkpoint.assert_called_once_with('test-only-checkpoint', strict=True)
        model.load_checkpoint.side_effect = RuntimeError('missing backbone tensor')
        with self.assertRaisesRegex(RuntimeError, 'missing backbone tensor'):
            self.ns['_load_model'](self.policy)


if __name__ == '__main__':
    main()
