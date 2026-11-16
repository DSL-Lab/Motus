"""Strict clean Place Can Basket episode membership for the A/C experiment."""
from pathlib import Path

import numpy as np


TASK = 'place_can_basket'
EPISODES = 50
TRAIN_EPISODES = 45
VAL_EPISODES = 5


def clean_episode_split(dataset_root, seed=0):
    """Use the converter's sorted paths and seeded 10% holdout before statistics."""
    task = Path(dataset_root).resolve() / 'clean' / TASK
    paths = sorted(path.resolve() for path in (task / 'qpos').glob('*.pt'))
    if len(paths) != EPISODES or len(set(paths)) != EPISODES:
        raise ValueError(f'Expected exactly 50 distinct clean {TASK} demonstrations in {task / "qpos"}; found {len(paths)}')
    episodes = {}
    for path in paths:
        expected = {
            'qpos_path': task / 'qpos' / path.name,
            'video_path': task / 'videos' / (path.stem + '.mp4'),
            'lang_path': task / 'umt5_wan' / path.name,
            'meta_path': task / 'metas' / (path.stem + '.txt'),
        }
        for file in expected.values():
            if not file.is_file() or not file.resolve().is_relative_to(task.resolve()):
                raise ValueError(f'Missing or out-of-task clean demonstration asset: {file}')
        episodes[str(path)] = {key: str(file.resolve()) for key, file in expected.items()}
    order = np.random.default_rng(seed).permutation(EPISODES)
    validation = {str(paths[index]) for index in order[:VAL_EPISODES]}
    training = {str(paths[index]) for index in order[VAL_EPISODES:]}
    return episodes, training, validation


def validate_dataset_split(config, normalization, scaling, manifest):
    """Bind windows and training statistics to the same complete 45/5 split."""
    dataset = config['dataset']
    if (dataset['task_mode'], dataset['task_name'], dataset['data_mode']) != ('single', TASK, 'clean'):
        raise ValueError('A/C data must use single-task clean place_can_basket demonstrations')
    seed = int(config['research'].get('split_seed', 0))
    episodes, training, validation = clean_episode_split(dataset['dataset_dir'], seed)
    sources = normalization['training_sources']
    norm_train = [str(Path(source['path']).resolve()) for source in sources]
    norm_val = [str(Path(path).resolve()) for path in normalization['validation_paths']]
    if (len(norm_train) != TRAIN_EPISODES or len(norm_val) != VAL_EPISODES
            or set(norm_train) != training or set(norm_val) != validation
            or normalization.get('seed') != seed or normalization.get('validation_fraction') != .1):
        raise ValueError('Normalization must use the standard seeded 45/5 clean demonstration split')
    source_keys = {(str(Path(source['path']).resolve()), source['action_sha256']) for source in sources}
    scaling_keys = [(str(Path(path).resolve()), digest) for path, digest in scaling['training_sources']]
    if len(scaling_keys) != TRAIN_EPISODES or set(scaling_keys) != source_keys:
        raise ValueError('Knot scaling must use exactly the same 45 training demonstrations')
    membership = {'train': training, 'validation': validation}
    covered = {'train': set(), 'validation': set()}
    for window in manifest['windows']:
        split = window['split']
        source = str(Path(window['qpos_path']).resolve())
        if split not in membership or source not in membership[split]:
            raise ValueError(f'Window violates the clean 45/5 split: {window["qpos_path"]} ({split})')
        if (window.get('task_name'), window.get('data_mode'), window.get('episode_name')) != (
                TASK, 'clean', Path(source).stem):
            raise ValueError('Window metadata must identify the clean place_can_basket episode')
        for key in ('video_path', 'lang_path'):
            if str(Path(window[key]).resolve()) != episodes[source][key]:
                raise ValueError(f'Window {key} does not match its clean demonstration')
        covered[split].add(source)
    if covered != membership:
        raise ValueError('Window manifest must cover all 45 training and all 5 validation demonstrations')
    return dict(task=TASK, data_mode='clean', episodes=EPISODES, train_episodes=TRAIN_EPISODES,
                validation_episodes=VAL_EPISODES, seed=seed, validation_fraction=.1,
                training_paths=sorted(training), validation_paths=sorted(validation))
