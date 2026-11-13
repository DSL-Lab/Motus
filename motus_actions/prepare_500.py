"""Allocated preparation of the pinned 500-demo dataset, reusing existing code.

The reference adapter's next-state-as-action convention is recorded explicitly;
physical timing/replay acceptance is not inferred from conversion success.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
from types import SimpleNamespace
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent/'WAM-RoboTwin'
REVISION = '3dc3b798668feb99ac61cc9086d84cbcc3d79186'
EXPECTED_SHA256 = 'e448a8e4f5049b8ab75c500dee55ef958c3f5326f8fc04630f2eca52ac1c871c'
ARCHIVE_ROOT = 'aloha-agilex_randomized_500'


def archive_members(archive):
    """Verify all member paths before extracting any data."""
    members = archive.infolist()
    names = set()
    episodes = set()
    for member in members:
        path = Path(member.filename)
        if (path.is_absolute() or '..' in path.parts or not path.parts or
                path.parts[0] != ARCHIVE_ROOT or stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError(f'unsafe/unexpected ZIP member: {member.filename}')
        if member.filename in names:
            raise ValueError(f'duplicate ZIP member: {member.filename}')
        names.add(member.filename)
        if path.suffix == '.hdf5':
            if len(path.parts) != 3 or path.parts[1] != 'data' or not path.stem.startswith('episode'):
                raise ValueError('unexpected raw HDF5 layout')
            episodes.add(int(path.stem.removeprefix('episode')))
    if episodes != set(range(500)):
        raise ValueError('archive must contain exactly episodes 0 through 499')
    if f'{ARCHIVE_ROOT}/seed.txt' not in names or f'{ARCHIVE_ROOT}/scene_info.json' not in names:
        raise ValueError('archive is missing the recorded scene/seed sidecars')
    return members


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('conversion/fitting/T5 encoding must run in a Slurm allocation')
    from .prepare import digest, prepare, prepare_language
    args.archive, args.work_dir, args.output = (p.resolve() for p in (args.archive, args.work_dir, args.output))
    if args.output.exists():
        raise FileExistsError(f'preserving existing cache/output: {args.output}')
    print('Verifying archive SHA-256', flush=True)
    if digest(args.archive) != EXPECTED_SHA256:
        raise ValueError('archive SHA-256 does not match the pinned public source')
    raw = args.work_dir/'raw'
    source = raw/'place_can_basket'/'demo_randomized'
    marker = source/'.extracted.json'
    if not marker.exists():
        temporary = args.work_dir/f'extraction-{os.environ["SLURM_JOB_ID"]}'
        temporary.mkdir(exist_ok=False)
        with zipfile.ZipFile(args.archive) as archive:
            members = archive_members(archive)
            free = shutil.disk_usage(args.work_dir).free
            if free < 3*sum(m.file_size for m in members):
                raise OSError('insufficient reported free space for raw, converted and cached data')
            archive.extractall(temporary, members=members)
        source.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            raise FileExistsError(f'incomplete raw extraction preserved: {source}')
        (temporary/ARCHIVE_ROOT).rename(source)
        marker.write_text(json.dumps(dict(sha256=EXPECTED_SHA256, revision=REVISION))+'\n')
    elif json.loads(marker.read_text()).get('sha256') != EXPECTED_SHA256:
        raise ValueError('existing raw extraction has a different source identity')
    seeds = [int(value) for value in (source/'seed.txt').read_text().split()]
    if len(seeds) != 500 or len(set(seeds)) != 500:
        raise ValueError('expected 500 distinct recorded scene seeds')
    # Reuse the verified reference adapter without editing its repository.
    converter = REFERENCE/'tools/convert_legacy_data.py'
    spec = importlib.util.spec_from_file_location('motus_reference_data_adapter', converter)
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    converted = args.work_dir/'converted'
    options = SimpleNamespace(input_root=raw, output_root=converted,
        env_cfg_type='aloha_agilex', frequency=None, expert_data_num=500,
        start_episode=0, overwrite=False, instruction_split='seen', instruction_index=0,
        instructions_format='array', same_index_action=False, keep_last_frame=False)
    adapter.convert_job(options, adapter.DatasetJob('place_can_basket', 'demo_randomized'))
    numerical = args.work_dir/'numerical'
    if not (numerical/'manifest.json').exists():
        prepare(converted/'demo_randomized', numerical, ['place_can_basket'], 500)
    language = args.output.parent/f'.{args.output.name}-language-{os.environ["SLURM_JOB_ID"]}'
    import torch
    with torch.inference_mode():
        prepare_language(numerical/'manifest.json', language, ROOT/'.runtime/model-metadata/Wan2.2-TI2V-5B')
    manifest_path = language/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest['episodes']:
        index = int(entry['episode_name'].rsplit('_', 1)[1])
        entry['seed'] = seeds[index]
    manifest['raw_dataset'] = dict(repo='TianxingChen/RoboTwin2.0', revision=REVISION,
        archive=str(args.archive), archive_sha256=EXPECTED_SHA256,
        raw_directory=str(source), seed_file_sha256=digest(source/'seed.txt'),
        reference_converter=str(converter), reference_converter_sha256=digest(converter),
        reference_commit='74f239e1a10f7b528d7488af956ee1e75f959415',
        action_alignment='next_state_as_action; source final frame dropped',
        physical_clock='unverified; converter frequency is not interpreted as Hz')
    manifest.pop('identity', None)
    manifest['identity'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    shutil.copy2(source/'seed.txt', language/'seed.txt')
    shutil.copy2(source/'scene_info.json', language/'scene_info.json')
    language.rename(args.output)
    print(f'PREPARED: {args.output / "manifest.json"}', flush=True)
    print('500 episodes, 450 train / 50 validation. Replay/repair and physical-clock gates remain pending.', flush=True)


if __name__ == '__main__':
    main()
