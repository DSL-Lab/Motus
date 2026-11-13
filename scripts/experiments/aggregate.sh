#!/usr/bin/env bash
# Small diagnostic report; never synthesizes policy-success records.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
python3 - "$repo" "${1:?output report directory required}" <<'PY'
import csv, json, re, statistics, subprocess, sys
from pathlib import Path
root, output = Path(sys.argv[1]), Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=False)
manifest_path = root/'outputs/pilot-cache-guard-20261111/manifest.json'
manifest = json.loads(manifest_path.read_text())
ids = sorted(set((root/'outputs/job_ids.txt').read_text().split()))
raw = subprocess.check_output(['sacct','-X','-P','-n','-j',','.join(ids),
    '--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES'], text=True)
(output/'slurm.psv').write_text(raw)
jobs = []
for row in csv.reader(raw.splitlines(), delimiter='|'):
    if len(row) < 5: continue
    gpu = re.search(r'gres/gpu=(\d+)', row[4])
    jobs.append(dict(job_id=row[0], state=row[1], exit_code=row[2],
        elapsed_s=int(row[3]), gpu_count=int(gpu.group(1)) if gpu else 0))
summary = dict(dataset_identity=manifest['identity'],
    train_episodes=len(manifest['split']['train']), validation_episodes=len(manifest['split']['validation']),
    episodes=len(manifest['episodes']), windows=sum(e['windows'] for e in manifest['episodes']),
    max_normalized_fit_error=max(e['fit_max_error'] for e in manifest['episodes']),
    median_global_knots=statistics.median(e['knots'] for e in manifest['episodes']),
    exclusions=manifest['excluded'], jobs=jobs,
    consumed_gpu_hours_as_of_report=sum(j['elapsed_s']*j['gpu_count'] for j in jobs)/3600,
    pilot_status='not submitted: recorded replay/controller gates unpassed',
    policy_evaluation_records=0)
smokes=[]
for path in sorted((root/'outputs').glob('model-smoke-*/results.json')):
    for result in json.loads(path.read_text()):
        smokes.append(dict(source=str(path), **result))
summary['measured_model_smokes']=smokes
overfits=[dict(source=str(p),**json.loads(p.read_text())) for p in sorted((root/'outputs').glob('tiny-*/result.json'))]
summary['tiny_overfit_diagnostics']=overfits
wrappers=[]
for path in sorted((root/'outputs').glob('wrapper-inference-*/results.json')):
    wrappers.extend(dict(source=str(path),**r) for r in json.loads(path.read_text()))
summary['offline_wrapper_diagnostics']=wrappers
(output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
lines=['# Overnight diagnostic report', '',
'No task-success, learned-policy latency, or final experiment claim is supported.', '',
'| Measured conversion quantity | Value |','|---|---:|',
*[f'| {k} | {summary[k]} |' for k in ('episodes','train_episodes','validation_episodes','windows','max_normalized_fit_error','median_global_knots')],
'',f"Excluded candidates: {summary['exclusions']}",'',
'| Job | State | Exit | Seconds |','|---|---|---|---:|',
*[f"| {j['job_id']} | {j['state']} | {j['exit_code']} | {j['elapsed_s']} |" for j in jobs],
'',f"Consumed GPU-hours at report: {summary['consumed_gpu_hours_as_of_report']:.4f} (includes retries; running jobs may accrue more).",'',
'Full model smoke measurements, if present, are single optimizer steps, not pilot learning or policy rollout evidence.',
'', '```json', json.dumps(smokes, indent=2), '```', '',
'Fixed-example overfit diagnostics (not pilot policy evidence):', '', '```json',
json.dumps([{k:v for k,v in r.items() if k!='losses'} for r in overfits],indent=2),'```','',
'Offline wrapper checks (not simulator episodes; first calls are not a warmed latency benchmark):', '',
'| Source | Configuration | Outcome | Knot repairs |', '|---|---|---|---:|',
*[f"| {Path(r['source']).parent.name} | {r['representation']} | {r['status']} | {r['counts']['knot_repairs']} |" for r in wrappers], '',
'Pilot/main/evaluation remain gated by missing recorded scene seeds and unverified fixed controller clock. '
'Exact unpublished SplineWAM alignment offset/rounding are explicit adaptations. '
'C36 action rows exceed A16; token or latency savings are unproven.']
(output/'report.md').write_text('\n'.join(lines)+'\n')
print(output/'report.md')
PY
