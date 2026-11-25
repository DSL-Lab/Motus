"""Strict clean Place Can Basket episode membership for the A/C experiment."""
from pathlib import Path

import numpy as np


TASK = 'place_can_basket'
TRAIN_EPISODES = 45
VAL_EPISODES = 5


def clean_episode_split(dataset_root, seed=0, *, train_episodes=TRAIN_EPISODES,
                        validation_episodes=VAL_EPISODES):
    """Split sorted episode paths before constructing windows or statistics."""
    if min(train_episodes, validation_episodes) < 1:
        raise ValueError('Both episode counts must be positive')
    total = train_episodes + validation_episodes
    task = Path(dataset_root).resolve() / 'clean' / TASK
    paths = sorted(path.resolve() for path in (task / 'qpos').glob('*.pt'))
    if len(paths) != total or len(set(paths)) != total:
        raise ValueError(f'Expected exactly {total} distinct clean {TASK} demonstrations in {task / "qpos"}; found {len(set(paths))}')
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
    order = np.random.default_rng(seed).permutation(total)
    validation = {str(paths[index]) for index in order[:validation_episodes]}
    training = {str(paths[index]) for index in order[validation_episodes:]}
    return episodes, training, validation


def validate_dataset_split(config, normalization, scaling, manifest):
    """Bind windows and both training statistics to the configured episode split."""
    dataset = config['dataset']
    if (dataset['task_mode'], dataset['task_name'], dataset['data_mode']) != ('single', TASK, 'clean'):
        raise ValueError('A/C data must use single-task clean place_can_basket demonstrations')
    research = config['research']
    seed = int(research.get('split_seed', 0))
    train_count = int(research.get('train_episodes', TRAIN_EPISODES))
    val_count = int(research.get('validation_episodes', VAL_EPISODES))
    total = train_count + val_count
    if dataset.get('max_episodes', total) != total:
        raise ValueError('dataset.max_episodes must match the configured episode split')
    episodes, training, validation = clean_episode_split(dataset['dataset_dir'], seed,
        train_episodes=train_count, validation_episodes=val_count)
    fraction = val_count / total
    sources = normalization['training_sources']
    norm_train = [str(Path(source['path']).resolve()) for source in sources]
    norm_val = [str(Path(path).resolve()) for path in normalization['validation_paths']]
    if (len(norm_train) != train_count or len(norm_val) != val_count
            or set(norm_train) != training or set(norm_val) != validation
            or normalization.get('seed') != seed or normalization.get('validation_fraction') != fraction):
        raise ValueError(f'Normalization must use the seeded {train_count}/{val_count} clean demonstration split')
    source_keys = {(str(Path(source['path']).resolve()), source['action_sha256']) for source in sources}
    scaling_keys = [(str(Path(path).resolve()), digest) for path, digest in scaling['training_sources']]
    if len(scaling_keys) != train_count or set(scaling_keys) != source_keys:
        raise ValueError(f'Knot scaling must use exactly the same {train_count} training demonstrations')
    membership = {'train': training, 'validation': validation}
    covered = {'train': set(), 'validation': set()}
    for window in manifest['windows']:
        split = window['split']
        source = str(Path(window['qpos_path']).resolve())
        if split not in membership or source not in membership[split]:
            raise ValueError(f'Window violates the clean {train_count}/{val_count} split: {window["qpos_path"]} ({split})')
        if (window.get('task_name'), window.get('data_mode'), window.get('episode_name')) != (
                TASK, 'clean', Path(source).stem):
            raise ValueError('Window metadata must identify the clean place_can_basket episode')
        for key in ('video_path', 'lang_path'):
            if str(Path(window[key]).resolve()) != episodes[source][key]:
                raise ValueError(f'Window {key} does not match its clean demonstration')
        covered[split].add(source)
    if covered != membership:
        raise ValueError(f'Window manifest must cover all {train_count} training and all {val_count} validation demonstrations')
    return dict(task=TASK, data_mode='clean', episodes=total, train_episodes=train_count,
                validation_episodes=val_count, seed=seed, validation_fraction=fraction,
                training_paths=sorted(training), validation_paths=sorted(validation))
