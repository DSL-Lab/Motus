"""Shared spline extraction, decoder resolution and MOTUS target integration."""
import ast
import contextlib
import importlib
import io
import json
from pathlib import Path
import unittest
import uuid

import numpy as np

from data.robotwin2 import adaptive_spline as adapter
from data.robotwin2 import research_windows
import robotwin_spline as shared
from robotwin_spline.local import extract_segment


class SplineIntegrationChecks(unittest.TestCase):
    def test_unpadded_extraction_is_model_independent(self):
        y = np.column_stack([np.sin(np.arange(100) / 3), np.cos(np.arange(100) / 4)])
        spline, _ = shared.fit_episode(y)
        for count in [8, 17, 36, 200]:
            for obs in [0., 21.5, 99.]:
                part = extract_segment(spline, obs, knot_count=count)
                self.assertEqual(len(part['knots']), len(part['control_points']) + 4)
                times = np.linspace(*part['valid_interval'], 101)
                from scipy.interpolate import BSpline
                actual = BSpline(part['knots'], part['control_points'], 3, extrapolate=False)(times)
                np.testing.assert_array_equal(actual, spline(times))

    def test_training_and_deployment_share_evaluator(self):
        root = Path(__file__).resolve().parents[1]
        contract_tree = ast.parse((root / 'utils/research.py').read_text())
        entry = next(n.value for n in ast.walk(contract_tree) if isinstance(n, ast.Constant)
                     and n.value == 'data.robotwin2.adaptive_spline.decode_local')
        module, function = entry.rsplit('.', 1)
        decoder = getattr(importlib.import_module(module), function)
        self.assertIs(research_windows.decode_local, decoder)
        self.assertIs(adapter.evaluate_local, shared.evaluate_local)
        # Resolve the actual stored decoder contract used by diagnostics.
        source = Path(__file__).resolve().parents[1] / 'train/research_diagnostic.py'
        tree = ast.parse(source.read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                   and n.module == 'data.robotwin2.adaptive_spline']
        self.assertTrue(any(a.name == 'decode_local' for n in imports for a in n.names))

    def test_actual_window_loader_packing_hash_and_split_checks(self):
        from robotwin_spline.convert import main
        root = Path(__file__).resolve().parents[1] / 'outputs/spline-tests' / uuid.uuid4().hex
        episodes = root / 'episodes'
        episodes.mkdir(parents=True)
        x = np.linspace(0, 12, 500)
        actions = np.column_stack([np.sin((j + 1) * x) for j in range(14)])
        for i in range(4):
            np.save(episodes / f'{i}.npy', actions)
        cache = root / 'fit.npz'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--episode-dir', str(episodes), '--output', str(cache)]), 0)
        commands, spline, meta = shared.read_cache(cache)
        # No image or torch deserialization is needed by load_windows itself;
        # these placeholder files satisfy only the original path preflight.
        for name in ['source.pt', 'video.mp4', 'lang.txt']:
            (root / name).touch()
        timeline = dict(status='verified', units='seconds', action_sha256=meta['source']['action_sha256'],
                        action_times=(np.arange(500) * .1).tolist(), camera_times=(np.arange(500) * .1).tolist())
        (root / 'timeline.json').write_text(json.dumps(timeline))
        contract = dict(normalization=meta['normalization'], knot_scaling=dict(
            scope='diagnostic-only', units='source_action_step', offset=0., scale=100.))
        window = dict(split='train', qpos_path=str(root / 'source.pt'), video_path=str(root / 'video.mp4'),
                      lang_path=str(root / 'lang.txt'), cache_path=str(cache),
                      timeline_path=str(root / 'timeline.json'), observation_index=0)
        manifest = root / 'windows.json'
        manifest.write_text(json.dumps(dict(schema='motus-fixed-real-windows-v1', windows=[window])))
        loaded = research_windows.load_windows(manifest, False, contract)[0]
        np.testing.assert_array_equal(research_windows.window_targets(loaded, 'spline')['actions'],
                                      adapter.extract_local(spline, 0, contract['knot_scaling'])['parameters'])
        np.testing.assert_array_equal(research_windows.window_targets(loaded, 'dense')['actions'],
                                      commands[np.arange(3, 49, 3)].astype(np.float32))
        window['split'] = 'validation'
        manifest.write_text(json.dumps(dict(schema='motus-fixed-real-windows-v1', windows=[window])))
        with self.assertRaisesRegex(ValueError, 'leaks'):
            research_windows.load_windows(manifest, True, contract)
        window['split'] = 'train'
        manifest.write_text(json.dumps(dict(schema='motus-fixed-real-windows-v1', windows=[window])))
        with np.load(cache) as f:
            contents = {key: f[key].copy() for key in f.files}
        contents['actions'][0, 0] += 1
        np.savez_compressed(cache, **contents)
        with self.assertRaisesRegex(ValueError, 'source identity'):
            research_windows.load_windows(manifest, False, contract)


if __name__ == '__main__':
    unittest.main()
