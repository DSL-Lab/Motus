"""Refit the prepared clean-50 commands and retain the original window membership."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from robotwin_spline import identity, read_cache
from robotwin_spline.convert import main as convert_spline
from robotwin_spline.sidecar import load_actions, source_identity

from data.robotwin2.adaptive_spline import fit_knot_time_scaling
from data.robotwin2.research_windows import load_windows
from .data import validate_dataset_split


def write_json(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def prepare(source, output):
    manifest = json.loads((source / 'windows.json').read_text())
    normalization = json.loads((source / 'normalization.json').read_text())
    if normalization.pop('identity') != identity(normalization):
        raise ValueError('Original action normalization identity mismatch')
    windows = manifest['windows']
    original_root = Path(windows[0]['qpos_path']).parents[4]

    def relocated(path):
        return source / Path(path).relative_to(original_root)

    for index, record in enumerate(normalization['training_sources']):
        path = relocated(record['path'])
        current = source_identity(path, load_actions(path))
        if current['action_sha256'] != record['action_sha256']:
            raise ValueError(f'Original commands changed: {path}')
        normalization['training_sources'][index] = current
    normalization['validation_paths'] = [str(relocated(path)) for path in normalization['validation_paths']]
    normalization['identity'] = identity(normalization)

    output.mkdir(parents=True, exist_ok=False)
    (output / 'sidecars').mkdir()
    (output / 'timelines').mkdir()
    normalization_path = output / 'normalization.json'
    write_json(normalization_path, normalization)

    caches, train_caches, fit_errors = {}, [], []
    for window in windows:
        for key in ('qpos_path', 'video_path', 'lang_path', 'cache_path', 'timeline_path'):
            window[key] = str(relocated(window[key]))
        window.pop('identity', None)
        original_cache = Path(window['cache_path'])
        if original_cache not in caches:
            qpos = Path(window['qpos_path'])
            timeline = json.loads(Path(window['timeline_path']).read_text())
            if timeline.pop('identity') != identity(timeline):
                raise ValueError(f'Original timeline identity mismatch: {qpos}')
            cache_path = output / 'sidecars' / original_cache.name
            timeline_path = output / 'timelines' / Path(window['timeline_path']).name
            if convert_spline(['--episode-dir', str(qpos.parent), '--episode', str(qpos),
                               '--output', str(cache_path), '--normalization', str(normalization_path),
                               '--insertion', 'max-residual']) != 0:
                raise ValueError(f'SplineWAM fit failed: {qpos}')
            _, _, meta = read_cache(cache_path)
            if meta['source']['action_sha256'] != timeline['action_sha256']:
                raise ValueError(f'Original commands changed: {qpos}')
            timeline['alignment']['qpos_path'] = str(qpos)
            timeline['identity'] = identity(timeline)
            write_json(timeline_path, timeline)
            caches[original_cache] = str(cache_path), str(timeline_path)
            fit_errors.append(meta['diagnostics']['max_abs_normalized_residual'])
            if window['split'] == 'train':
                train_caches.append(cache_path)
        window['cache_path'], window['timeline_path'] = caches[original_cache]

    scaling = fit_knot_time_scaling(train_caches)
    write_json(output / 'knot_scaling.json', scaling)
    path = output / 'windows.json'
    write_json(path, manifest)
    config = dict(dataset=dict(dataset_dir=str(source / 'dataset'), task_mode='single',
                               task_name='place_can_basket', data_mode='clean'), research=dict(split_seed=0))
    validate_dataset_split(config, normalization, scaling, manifest)
    contract = dict(normalization=normalization, knot_scaling=scaling, executed_prefix=dict(row_budget=36))
    loaded = [window for validation in (False, True) for window in load_windows(path, validation, contract)]
    summary = dict(windows=dict(Counter(window['split'] for window in loaded)), sidecars=len(caches),
                   original_manifest_sha256=hashlib.sha256((source / 'windows.json').read_bytes()).hexdigest(),
                   max_normalized_fit_residual=max(fit_errors),
                   windows_without_future_video=sum(not any(window['_selection']['proposed_latent_loss_mask'])
                                                    for window in loaded))
    write_json(output / 'preparation_report.json', summary)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Prepared original clean-50 corpus')
    parser.add_argument('--output', type=Path, required=True, help='New directory for corrected artifacts')
    args = parser.parse_args()
    prepare(args.source.resolve(strict=True), args.output.resolve())


if __name__ == '__main__':
    main()
