"""Small recorded-schema fixtures for the clean-50 preprocessing bridge."""
import json
import sys
from pathlib import Path
import tempfile
import unittest

import cv2
import h5py
import numpy as np
import torch

WAM_SCRIPTS = Path(__file__).resolve().parents[4] / 'WAM-RoboTwin/scripts'
sys.path.insert(0, str(WAM_SCRIPTS))
import prepare_motus_clean50 as preparation
from prepare_motus_clean50 import (
    inspect_sources, convert_recording, compose_rgb, make_timeline, write_json)
from robotwin_spline import identity


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'raw/data').mkdir(parents=True)
        (self.root / 'raw/instruction').mkdir()
        self.actions = np.arange(80*14, dtype=np.float32).reshape(80, 14) / 100
        self.path = self.root / 'raw/data/episode_0000000.hdf5'
        with h5py.File(self.path, 'w') as file:
            file.create_dataset('data_format_version', data='v1.0', dtype=h5py.string_dtype())
            file.create_dataset('action/joint_states', data=self.actions)
            for name, start, end in [('left_arm_joint_states', 0, 6), ('left_ee_joint_states', 6, 7),
                                     ('right_arm_joint_states', 7, 13), ('right_ee_joint_states', 13, 14)]:
                file.create_dataset('action/' + name, data=self.actions[:, start:end])
            file.create_dataset('state/joint_states', data=self.actions - 1)
            for camera, color in [('cam_head', [220, 15, 25]), ('cam_left_wrist', [40, 180, 60]),
                                  ('cam_right_wrist', [70, 80, 210])]:
                image = np.full((16, 16, 3), color, np.uint8)
                success, encoded = cv2.imencode('.jpg', image)
                self.assertTrue(success)
                file.create_dataset(f'vision/{camera}/colors', data=np.array([encoded.tobytes()]*80, dtype=f'S{len(encoded)}'))
        import os
        for index in range(50):
            path = self.root / 'raw/data' / f'episode_{index:07d}.hdf5'
            if index:
                os.link(self.path, path)
            (self.root / 'raw/instruction' / f'episode_{index:07d}.json').write_text(
                json.dumps({'seen': ['Place the can in the basket.'], 'unseen': ['Held-out phrasing.']}))

    def test_source_inspection_requires_50_aligned_recordings_and_seen_instruction(self):
        records = inspect_sources(self.root / 'raw')
        self.assertEqual(len(records), 50)
        self.assertEqual(records[0]['rows'], 80)
        self.assertIn('Place the can', records[0]['instruction'])
        self.assertNotIn('Held-out', records[0]['instruction'])
        (self.root / 'raw/data/episode_0000049.hdf5').unlink()
        with self.assertRaisesRegex(ValueError, 'exactly 50'):
            inspect_sources(self.root / 'raw')

    def test_conversion_preserves_action_rows_and_camera_layout(self):
        record = inspect_sources(self.root / 'raw')[0]
        task = self.root / 'output/clean/place_can_basket'
        qpos = convert_recording(record, task)
        torch.testing.assert_close(torch.load(qpos, weights_only=True), torch.from_numpy(self.actions), rtol=0, atol=0)
        capture = cv2.VideoCapture(str(task / 'videos/episode_0000000.mp4'))
        self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 80)
        success, bgr = capture.read()
        capture.release()
        self.assertTrue(success)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self.assertEqual(rgb.shape, (24, 16, 3))
        np.testing.assert_allclose(rgb[4, 4], [220, 15, 25], atol=8)
        np.testing.assert_allclose(rgb[20, 3], [40, 180, 60], atol=8)
        np.testing.assert_allclose(rgb[20, 12], [70, 80, 210], atol=8)
        self.assertEqual(convert_recording(record, task), qpos)

    def test_rejects_dropped_camera_rows(self):
        with h5py.File(self.path, 'a') as file:
            del file['vision/cam_head/colors']
            file.create_dataset('vision/cam_head/colors', data=np.zeros(79, dtype='S10'))
        with self.assertRaisesRegex(ValueError, 'row mismatch'):
            inspect_sources(self.root / 'raw')

    def test_full_50_episode_pipeline_with_synthetic_language_embeddings(self):
        import contextlib
        import io
        from unittest.mock import patch
        x = np.linspace(0, 12, 500)
        actions = np.column_stack([np.sin((index+1)*x) for index in range(14)]).astype(np.float32)
        # The 50 source paths share this fixture inode; update all records together.
        with h5py.File(self.path, 'a') as file:
            for name in ['action/joint_states', 'state/joint_states']:
                del file[name]
                file.create_dataset(name, data=actions)
            for name, start, end in [('left_arm_joint_states', 0, 6), ('left_ee_joint_states', 6, 7),
                                     ('right_arm_joint_states', 7, 13), ('right_ee_joint_states', 13, 14)]:
                del file['action/' + name]
                file.create_dataset('action/' + name, data=actions[:, start:end])
            for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                encoded = file[f'vision/{camera}/colors'][0]
                del file[f'vision/{camera}/colors']
                file.create_dataset(f'vision/{camera}/colors', data=np.array([encoded]*len(actions)))
        records = inspect_sources(self.root / 'raw')
        def fake_language(records, task, assets):
            for record in records:
                torch.save([torch.zeros(2,4096)], task / 'umt5_wan' / (Path(record['path']).stem + '.pt'))
        output = self.root / 'prepared'
        with patch.object(preparation, 'encode_language', side_effect=fake_language), contextlib.redirect_stdout(io.StringIO()):
            preparation.prepare(records, output, self.root / 'unused-model-assets')
        report = json.loads((output / 'preparation_report.json').read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(report['dataset_split']['train_episodes'], 45)
        self.assertEqual(report['dataset_split']['validation_episodes'], 5)
        self.assertGreaterEqual(report['train_windows'], 45)
        self.assertGreaterEqual(report['validation_windows'], 5)
        manifest = json.loads((output / 'windows.json').read_text())
        self.assertEqual(len({window['qpos_path'] for window in manifest['windows']}), 50)
        self.assertEqual(len(list((output / 'sidecars').glob('*.npz'))), 50)

    def test_fresh_cleanup_removes_only_the_generated_clean50_output(self):
        from prepare_motus_clean50 import reset_output
        target = self.root / 'outputs/ac_spline_pilot/clean50'
        target.mkdir(parents=True)
        (target / 'failed-artifact').write_text('partial')
        reset_output(target, self.root)
        self.assertEqual(list(target.iterdir()), [])
        raw = self.root / 'raw'
        with self.assertRaisesRegex(ValueError, 'Refusing cleanup'):
            reset_output(raw, self.root)
        target.rmdir()
        target.symlink_to(raw, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Refusing cleanup'):
            reset_output(target, self.root)
        self.assertTrue(self.path.exists())

    def test_official_wan_config_metadata_is_not_forwarded_to_constructor(self):
        import logging
        import types
        import typing
        from experiments.ac_spline_pilot.tests.test_research import definitions
        class Backbone(torch.nn.Module):
            def __init__(self, **settings):
                super().__init__()
                self.settings = settings
                self.weight = torch.nn.Parameter(torch.zeros(1))
        namespace = dict(torch=torch, nn=torch.nn, WanModel=Backbone,
                         Wan2_2_VAE=lambda **kwargs: types.SimpleNamespace(),
                         resolve_device=lambda value: torch.device('cpu'),
                         logger=logging.getLogger(__name__),
                         **{key: getattr(typing, key) for key in ('Dict', 'List', 'Optional', 'Any')})
        definitions('models/wan_model.py', {'WanVideoModel'}, namespace)
        model = namespace['WanVideoModel'](
            dict(dim=4, _class_name='WanModel', _diffusers_version='metadata'), 'unused', device='cpu')
        self.assertEqual(model.wan_model.settings, {'dim': 4})

    def test_timeline_uses_recorded_rows_without_claiming_physical_seconds(self):
        record = inspect_sources(self.root / 'raw')[0]
        timeline = make_timeline(record, self.root / 'actions.pt')
        self.assertEqual(timeline['units'], 'source_action_step')
        self.assertEqual(timeline['alignment']['physical_clock'], 'unknown')
        self.assertEqual(timeline['action_times'], list(range(80)))
        self.assertEqual(timeline['camera_times'], list(range(80)))
        self.assertEqual(timeline['identity'], identity({key: value for key, value in timeline.items() if key != 'identity'}))
        output = self.root / 'timeline.json'
        write_json(output, timeline)
        write_json(output, timeline)
        with self.assertRaisesRegex(ValueError, 'differs'):
            write_json(output, dict(timeline, status='modified'))


if __name__ == '__main__':
    unittest.main()
