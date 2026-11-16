"""Require the complete clean 50-demo corpus and canonical 45/5 statistics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from robotwin_spline import identity

from experiments.ac_spline_pilot.data import clean_episode_split, validate_dataset_split
from experiments.ac_spline_pilot.research import load_contract


class DataSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        task = self.root / 'clean/place_can_basket'
        for folder, suffix in [('qpos', '.pt'), ('videos', '.mp4'), ('umt5_wan', '.pt'), ('metas', '.txt')]:
            (task / folder).mkdir(parents=True)
            for index in range(50):
                (task / folder / f'episode{index}{suffix}').touch()
        self.episodes, self.training, self.validation = clean_episode_split(self.root)
        self.config = dict(dataset=dict(dataset_dir=str(self.root), task_mode='single',
                                       task_name='place_can_basket', data_mode='clean'),
                           common=dict(num_video_frames=8, video_action_freq_ratio=2),
                           research=dict(split_seed=0, representation='spline', diagnostic=dict(windows=4),
                                         lora=dict(rank=64, alpha=128., dropout=0.,
                                                   target_modules=['video_model.wan_model.*', 'action_expert.*'])))
        self.normalization = dict(scope='training-only', seed=0, validation_fraction=.1,
                                  training_sources=[dict(path=path, action_sha256=f'hash-{Path(path).stem}')
                                                    for path in sorted(self.training)],
                                  validation_paths=sorted(self.validation), low=[-1.]*14, high=[1.]*14)
        self.normalization['identity'] = identity(self.normalization)
        self.scaling = dict(scope='training-only', units='source_action_step', offset=0., scale=100.,
                            normalization_identity=self.normalization['identity'],
                            training_sources=[[source['path'], source['action_sha256']]
                                              for source in self.normalization['training_sources']])
        self.scaling['identity'] = identity(self.scaling)
        self.manifest = dict(schema='motus-fixed-real-windows-v1', windows=[
            dict(split='train' if path in self.training else 'validation', qpos_path=path,
                 video_path=files['video_path'], lang_path=files['lang_path'], observation_index=0,
                 task_name='place_can_basket', data_mode='clean', episode_name=Path(path).stem)
            for path, files in self.episodes.items()])

    def validate(self):
        return validate_dataset_split(self.config, self.normalization, self.scaling, self.manifest)

    def test_complete_split_matches_converter_order_and_is_disjoint(self):
        result = self.validate()
        paths = sorted(self.episodes)
        order = np.random.default_rng(0).permutation(50)
        self.assertEqual(set(result['validation_paths']), {paths[index] for index in order[:5]})
        self.assertEqual((len(result['training_paths']), len(result['validation_paths'])), (45, 5))
        self.assertFalse(self.training & self.validation)
        self.assertEqual(self.validate(), result)

    def test_contract_accepts_all_50_episodes_while_preview_uses_four_windows(self):
        for name, data in [('normalization', self.normalization), ('scaling', self.scaling), ('windows', self.manifest)]:
            (self.root / f'{name}.json').write_text(json.dumps(data))
        self.config['research'].update(normalization_path=str(self.root / 'normalization.json'),
                                       knot_scaling_path=str(self.root / 'scaling.json'),
                                       window_manifest=str(self.root / 'windows.json'))
        contract = load_contract(self.config)
        self.assertEqual(contract['dataset_split']['train_episodes'], 45)
        self.assertEqual(contract['dataset_split']['validation_episodes'], 5)
        self.assertEqual(contract['packed_rows'], 36)

    def test_rejects_missing_extra_or_incomplete_demonstrations(self):
        qpos = self.root / 'clean/place_can_basket/qpos'
        extra = qpos / 'episode50.pt'
        extra.touch()
        with self.assertRaisesRegex(ValueError, 'exactly 50'):
            self.validate()
        extra.unlink()
        original = qpos / 'episode0.pt'
        original.unlink()
        with self.assertRaisesRegex(ValueError, 'exactly 50'):
            self.validate()
        original.touch()
        (self.root / 'clean/place_can_basket/videos/episode0.mp4').unlink()
        with self.assertRaisesRegex(ValueError, 'asset'):
            self.validate()

    def test_rejects_randomized_or_other_task_windows_and_assets(self):
        window = self.manifest['windows'][0]
        original = copy.deepcopy(window)
        for key, value in [('data_mode', 'randomized'), ('task_name', 'other_task'),
                           ('qpos_path', str(self.root / 'randomized/place_can_basket/qpos/episode0.pt')),
                           ('video_path', str(self.root / 'randomized/place_can_basket/videos/episode0.mp4'))]:
            with self.subTest(key=key):
                window.update(original)
                window[key] = value
                with self.assertRaises(ValueError):
                    self.validate()

    def test_rejects_missing_coverage_and_validation_leakage(self):
        original = copy.deepcopy(self.manifest)
        self.manifest['windows'].pop()
        with self.assertRaisesRegex(ValueError, 'cover all 45'):
            self.validate()
        self.manifest = original
        train = next(window for window in self.manifest['windows'] if window['split'] == 'train')
        train['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'violates'):
            self.validate()

    def test_rejects_wrong_normalization_or_scaling_membership(self):
        original = copy.deepcopy(self.normalization)
        self.normalization['training_sources'][0]['path'] = next(iter(self.validation))
        with self.assertRaisesRegex(ValueError, 'Normalization'):
            self.validate()
        self.normalization = original
        self.scaling['training_sources'].pop()
        with self.assertRaisesRegex(ValueError, 'Knot scaling'):
            self.validate()


if __name__ == '__main__':
    unittest.main()
