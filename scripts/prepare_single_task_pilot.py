"""CPU-only readiness audit and frozen candidate configuration. Never submits jobs.

The existing trainer deliberately refuses a full pilot until real prerequisites
pass. This utility preserves that guard and labels generated jobs as blocked.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ENV = ('MOTUS_WINDOWS', 'MOTUS_NORMALIZATION', 'MOTUS_KNOT_SCALING',
                'MOTUS_DATASET_ROOT', 'MOTUS_BASE_CHECKPOINT', 'MOTUS_WAN', 'MOTUS_VAE', 'MOTUS_VLM')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def source_snapshot(repo, destination):
    """Copy tracked and nonignored source, never external symlinks or submodules."""
    names = subprocess.check_output(
        ['git', '-C', str(repo), 'ls-files', '--cached', '--others', '--exclude-standard', '-z']
    ).decode('utf-8').split('\0')
    hashes = {}
    for name in sorted(set(names) - {''}):
        source = repo / name
        if source.suffix.lower() not in {'.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.md', '.txt'}:
            continue
        if source.is_symlink() or not source.is_file():
            continue
        if not source.resolve().is_relative_to(repo.resolve()):
            raise ValueError(f'Source escapes repository: {source}')
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    return dict(head=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip(),
                status=subprocess.check_output(['git', '-C', str(repo), 'status', '--short']).decode(),
                sha256=hashes)


def audit(robotwin):
    packages = {}
    for name in ('torch', 'numpy', 'scipy', 'sapien', 'decord', 'accelerate', 'transformers', 'omegaconf', 'yaml'):
        available = importlib.util.find_spec(name) is not None
        try:
            version = importlib.metadata.version('PyYAML' if name == 'yaml' else name) if available else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[name] = dict(importable=available, version=version)
    cuda = dict(available=False, devices=0)
    if packages['torch']['importable']:
        import torch
        cuda = dict(available=torch.cuda.is_available(), devices=torch.cuda.device_count(),
                    torch_version=torch.__version__)
    inputs = {name: dict(value=os.environ.get(name), exists=bool(os.environ.get(name)) and Path(os.environ[name]).exists())
              for name in ARTIFACT_ENV}
    module = robotwin / 'envs/place_can_basket.py'
    identifier_verified = module.is_file() and any(
        isinstance(n, ast.ClassDef) and n.name == 'place_can_basket'
        for n in ast.parse(module.read_text(encoding='utf-8')).body)
    dataset = os.environ.get('MOTUS_DATASET_ROOT')
    complete = []
    if dataset:
        task = Path(dataset) / 'clean/place_can_basket'
        complete = [p.stem for p in sorted((task / 'qpos').glob('*.pt'))
                    if all((task / sub / (p.stem + ext)).is_file()
                           for sub, ext in [('videos', '.mp4'), ('umt5_wan', '.pt'), ('metas', '.txt')])]
    blockers = []
    if not cuda['available']:
        blockers.append('No CUDA device in the selected interpreter')
    if any(not packages[n]['importable'] for n in ('sapien', 'decord', 'transformers', 'omegaconf')):
        blockers.append('Incomplete full-model/simulator environment')
    if any(not item['exists'] for item in inputs.values()):
        blockers.append('Required real data, statistics, clocks and/or model assets have not been supplied')
    if len(complete) < 50:
        blockers.append('50 complete Place Can Basket demonstrations not located')
    blockers += [
        'Original checkpoint provenance, strict full-model load and closed-loop deployment unexecuted',
        'Real conversion, local coverage/video audit and matched ground-truth simulator replay unexecuted',
        'Full-model GPU gradients/VAE masks, varied-noise tiny learning and fresh reload unexecuted',
        'Trainer enforces tiny-only phase and at most four real training windows; pilot unsupported',
        'Installed evaluator uses XPolicyLab; bundled MOTUS script expects absent script/eval_policy.py',
        'take_action(qpos) uses variable-length TOPP; a matched fixed control-step execution bridge is unverified',
        'Research adapter/spline deployment bridge and requested rollout timing instrumentation unimplemented',
        'No verified Slurm partition, account, GPU type, environment setup or walltime limits supplied',
        'No explicit session compute budget available to this preparation',
    ]
    return dict(schema='motus-pilot-readiness-v1', captured_at=datetime.now(timezone.utc).isoformat(),
                interpreter=sys.executable, packages=packages, cuda=cuda,
                commands={n: shutil.which(n) for n in ('sbatch', 'srun', 'nvidia-smi', 'ssh')},
                environment=inputs, verified_task='place_can_basket' if identifier_verified else None,
                complete_episode_ids=complete, episode_contents_validated=False,
                ready=False, blockers=blockers, jobs_submitted=[], requested_gpu_hours=None,
                budget_status='Cannot size exact jobs until target environment and prerequisites are verified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['place_can_basket'], default='place_can_basket')
    parser.add_argument('--representation', choices=['A', 'C', 'both'], default='both')
    parser.add_argument('--seed', type=int, choices=[0], default=0)
    parser.add_argument('--checkpoint', default=os.environ.get('MOTUS_BASE_CHECKPOINT'))
    parser.add_argument('--updates', type=int, choices=[1000], default=1000)
    parser.add_argument('--robotwin-root', type=Path, default=ROOT.parent / 'WAM-RoboTwin')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to((ROOT / 'outputs').resolve()):
        parser.error('Preparation outputs must be inside Motus/outputs')
    # Exclusive creation makes repeated preparation with the same output fail;
    # there is no sbatch call and no ambiguous automatic submission retry.
    out.mkdir(parents=True, exist_ok=False)
    readiness = audit(args.robotwin_root.resolve())
    dump(out / 'readiness.json', readiness)
    snapshots = {}
    for label, repo in [('Motus', ROOT), ('WAM-RoboTwin', args.robotwin_root.resolve())]:
        snapshots[label] = source_snapshot(repo, out / 'source' / label)
    variants = ['A', 'C'] if args.representation == 'both' else [args.representation]
    candidate_jobs = []
    for variant in variants:
        # JSON is valid YAML, and the existing recursive loader reads it.
        candidate = dict(base_config=f'robotwin_research_{variant}.yaml',
                         research=dict(phase='pilot', seed=args.seed),
                         dataset=dict(task_name=args.task),
                         training=dict(batch_size=1, gradient_accumulation_steps=16,
                                       max_steps=args.updates, learning_rate=1e-4, wan_learning_rate=1e-4,
                                       warmup_steps=200, cycle_length=5000000, f_max=.99, f_min=.4),
                         system=dict(save_interval=args.updates),
                         logging=dict(run_name='pilot-seed-0', report_to='none'))
        if args.checkpoint:
            candidate['finetune'] = dict(checkpoint_path=args.checkpoint)
        config = out / 'source/Motus/configs' / f'pilot_{variant}.json'
        dump(config, candidate)
        candidate_jobs.append(dict(name=f'{args.task}-{variant}-seed{args.seed}', status='blocked-not-submitted',
                                   representation=variant, config=str(config),
                                   command=['python', 'train/train.py', '--config', f'configs/pilot_{variant}.json',
                                            '--checkpoint_dir', str(out / 'runs' / variant), '--report_to', 'none'],
                                   cwd=str(out / 'source/Motus'), gpu_count=None, gpu_type=None,
                                   time_limit_hours=None, gpu_hours=None,
                                   warning='Current trainer intentionally rejects this pilot phase; not a validated job'))
    # Include generated configs in the checksum contract too.
    for path in (out / 'source/Motus/configs').glob('pilot_*.json'):
        snapshots['Motus']['sha256'][str(path.relative_to(out / 'source/Motus')).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
    dump(out / 'source-identities.json', snapshots)
    dump(out / 'pending-jobs.json', dict(schema='motus-pilot-preparation-v1', task=args.task,
         training_episode_count=45, validation_episode_count=5, split_status='proposed; no real split established',
         development_scene_count=20, development_scene_seeds=None, final_scene_seeds=None,
         scene_status='Must establish reset semantics and training-scene overlap before freezing seeds',
         original_deployment=dict(status='blocked-not-submitted', checkpoint=args.checkpoint, trials_completed=0),
         jobs=candidate_jobs, job_ids=[], total_requested_gpu_hours=None,
         submission_enabled=False, prerequisite_report='readiness.json'))
    print(json.dumps(dict(output=str(out), ready=False, blockers=len(readiness['blockers']),
                          jobs_submitted=0, total_requested_gpu_hours=None)))


if __name__ == '__main__':
    main()
