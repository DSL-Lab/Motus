"""Exercise the batch launcher's sequencing and container environment without Slurm."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/submit_full_pilot.sh'


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for directory in ('train', 'scripts/cluster', 'bin', 'dataset/clean/place_can_basket/qpos'):
            (self.root / directory).mkdir(parents=True)
        (self.root / 'train/train.py').touch()
        wrapper = self.root / 'scripts/cluster/environment.sh'
        wrapper.write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ "$1" == --gpu ]]
shift
# Simulate Apptainer --cleanenv while keeping the test recorder in PATH.
exec env -i PATH="$PATH" TEST_RECORD="$TEST_RECORD" TEST_FAIL="${TEST_FAIL:-}" TEST_SIM_FAIL_A="${TEST_SIM_FAIL_A:-}" TEST_OMIT_CHECKPOINT="${TEST_OMIT_CHECKPOINT:-}" "$@"
''')
        srun = self.root / 'bin/srun'
        srun.write_text('''#!/usr/bin/env bash
set -euo pipefail
while [[ "$1" == --* ]]; do shift; done
exec "$@"
''')
        srun.chmod(0o755)
        python = self.root / 'bin/python'
        python.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
arguments = sys.argv[1:]
with open(os.environ['TEST_RECORD'], 'a') as output:
    output.write(json.dumps(dict(args=arguments, env={key: value for key, value in os.environ.items()
                                                    if key.startswith('MOTUS_') or key == 'OMP_NUM_THREADS'})) + '\\n')
if os.environ.get('TEST_FAIL'):
    sys.exit(7)
if 'experiments.ac_spline_pilot.trainer' in arguments and not os.environ.get('TEST_OMIT_CHECKPOINT'):
    variant = 'A' if 'robotwin_pilot_A.yaml' in ' '.join(arguments) else 'C'
    root = Path(arguments[arguments.index('--checkpoint_dir')+1])
    checkpoint = root / ('robotwin_pilot_'+variant) / 'pilot-seed-0/checkpoint_step_17'
    checkpoint.mkdir(parents=True)
    (checkpoint/'research_adapter.pt').touch()
    (checkpoint/'research_config.json').write_text('{}')
    (checkpoint.parent/'final_checkpoint.txt').write_text('checkpoint_step_17\\n')
if 'experiments.ac_spline_pilot.simulator' in arguments and os.environ.get('TEST_SIM_FAIL_A'):
    if arguments[arguments.index('--output')+1].endswith('/A'):
        sys.exit(9)
''')
        python.chmod(0o755)
        sbatch = self.root / 'bin/sbatch'
        sbatch.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['TEST_SBATCH_RECORD'], 'a') as output:
    output.write(json.dumps(dict(args=sys.argv[1:], fresh=os.environ.get('MOTUS_PREP_FRESH'))) + '\\n')
print('123;cluster' if '--parsable' in sys.argv else 'Submitted batch job 456')
""")
        sbatch.chmod(0o755)
        self.env = dict(os.environ, PATH=str(self.root / 'bin') + os.pathsep + os.environ['PATH'],
                        MOTUS_REPO_ROOT=str(self.root), SLURM_JOB_ID='test-123', SLURM_CPUS_PER_TASK='8',
                        MOTUS_DATASET_ROOT=str(self.root / 'dataset'), TEST_RECORD=str(self.root / 'calls.jsonl'),
                        TEST_SBATCH_RECORD=str(self.root / 'sbatch.jsonl'), MOTUS_EVAL_EPISODES='7', MOTUS_EVAL_SEED='2')
        for name in ('MOTUS_WINDOWS', 'MOTUS_NORMALIZATION', 'MOTUS_KNOT_SCALING',
                     'MOTUS_BASE_CHECKPOINT', 'MOTUS_WAN', 'MOTUS_VAE', 'MOTUS_VLM'):
            path = self.root / name
            path.touch()
            self.env[name] = str(path)
        # Keep inherited user output overrides from influencing this isolated test.
        self.env['MOTUS_PILOT_OUTPUT'] = str(self.root / 'outputs/run')

    def run_script(self, *arguments):
        return subprocess.run(['bash', str(SCRIPT), *arguments], cwd=self.root, env=self.env,
                              capture_output=True, text=True)

    def test_sequential_variants_forward_assets_through_clean_environment(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in (self.root / 'calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(calls), 4)
        for variant, call in zip(('A', 'C'), calls[:2]):
            self.assertEqual(call['args'][:3], ['-u', '-m', 'experiments.ac_spline_pilot.trainer'])
            self.assertNotIn('--factory', call['args'])
            self.assertNotIn('--task', call['args'])
            self.assertIn(f'experiments/ac_spline_pilot/configs/robotwin_pilot_{variant}.yaml', call['args'])
            self.assertEqual(call['args'][-2:], ['--report_to', 'none'])
            self.assertEqual(call['env']['OMP_NUM_THREADS'], '8')
            for name in ('MOTUS_DATASET_ROOT', 'MOTUS_WINDOWS', 'MOTUS_NORMALIZATION', 'MOTUS_KNOT_SCALING',
                         'MOTUS_BASE_CHECKPOINT', 'MOTUS_WAN', 'MOTUS_VAE', 'MOTUS_VLM'):
                self.assertEqual(call['env'][name], self.env[name])

    def test_simulator_loads_current_run_checkpoints_and_matched_cases(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line)['args'] for line in (self.root/'calls.jsonl').read_text().splitlines()]
        for variant, call in zip(('A','C'), calls[2:]):
            self.assertEqual(call[:3], ['-u','-m','experiments.ac_spline_pilot.simulator'])
            self.assertEqual(call[call.index('--checkpoint')+1], str(self.root/f'outputs/run/{variant}/robotwin_pilot_{variant}/pilot-seed-0/checkpoint_step_17'))
            self.assertEqual(call[call.index('--cases')+1], str(self.root/'outputs/run/simulator/cases.json'))
            self.assertEqual(call[call.index('--episodes')+1], '7')
            self.assertEqual(call[call.index('--seed')+1], '2')
            self.assertTrue((self.root/f'outputs/run/logs/simulator_{variant}.log').is_file())

    def test_submit_prepares_then_uses_afterok_without_running_training_locally(self):
        self.env.pop('SLURM_JOB_ID')
        result = self.run_script('--submit','--task','place_can_basket')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        submissions = [json.loads(line) for line in (self.root/'sbatch.jsonl').read_text().splitlines()]
        self.assertEqual(len(submissions), 2)
        self.assertEqual(submissions[0]['fresh'], '0')
        self.assertIn('submit_prepare_clean50.sh', ' '.join(submissions[0]['args']))
        self.assertIn('--dependency=afterok:123', submissions[1]['args'])
        self.assertEqual(submissions[1]['args'][-2:], ['--task','place_can_basket'])
        self.assertFalse((self.root/'calls.jsonl').exists())

    def test_missing_final_checkpoint_stops_before_other_training_or_evaluation(self):
        self.env['TEST_OMIT_CHECKPOINT']='1'
        result=self.run_script()
        self.assertEqual(result.returncode,2,result.stdout+result.stderr)
        self.assertIn('did not publish its final checkpoint',result.stderr)
        self.assertEqual(len((self.root/'calls.jsonl').read_text().splitlines()),1)

    def test_evaluation_failure_still_attempts_the_other_policy(self):
        self.env['TEST_SIM_FAIL_A']='1'
        result=self.run_script()
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertEqual(len((self.root/'calls.jsonl').read_text().splitlines()),4)

    def test_variant_failure_stops_before_launching_c(self):
        self.env['TEST_FAIL'] = '1'
        result = self.run_script()
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertEqual(len((self.root / 'calls.jsonl').read_text().splitlines()), 1)

    def test_path_check_runs_without_slurm_and_never_launches_training(self):
        self.env.pop('SLURM_JOB_ID')
        result = self.run_script('--check')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('path-only check', result.stdout)
        self.assertFalse((self.root / 'calls.jsonl').exists())

    def test_training_dry_run_prints_both_commands_without_execution_or_outputs(self):
        self.env.pop('SLURM_JOB_ID')
        result = self.run_script('--dry-run')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for variant in ('A', 'C'):
            self.assertIn(f'robotwin_pilot_{variant}.yaml', result.stdout)
        self.assertIn('python -u -m experiments.ac_spline_pilot.trainer', result.stdout)
        self.assertIn('experiments.ac_spline_pilot.simulator', result.stdout)
        self.assertIn('afterok:<PREPARATION_JOB_ID>', result.stdout)
        self.assertFalse((self.root / 'calls.jsonl').exists())
        self.assertFalse((self.root / 'outputs').exists())

    def test_preparation_dry_run_preserves_inputs_and_explicit_resume_mode(self):
        self.env.pop('SLURM_JOB_ID')
        script = SCRIPT.with_name('submit_prepare_clean50.sh')
        for fresh in ('0', '1'):
            self.env['MOTUS_PREP_FRESH'] = fresh
            result = subprocess.run(['bash', str(script), '--dry-run'], cwd=self.root,
                                    env=self.env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('prepare_motus_clean50.py', result.stdout)
            self.assertEqual('--fresh' in result.stdout, fresh == '1')
        self.assertFalse((self.root / 'calls.jsonl').exists())
        self.assertFalse((self.root / 'outputs').exists())

    def test_path_check_lists_every_missing_artifact(self):
        self.env.pop('SLURM_JOB_ID')
        self.env['MOTUS_WINDOWS'] = str(self.root / 'missing-windows.json')
        self.env['MOTUS_NORMALIZATION'] = str(self.root / 'missing-normalization.json')
        result = self.run_script('--check')
        self.assertEqual(result.returncode, 2)
        self.assertIn('Missing MOTUS_WINDOWS', result.stderr)
        self.assertIn('Missing MOTUS_NORMALIZATION', result.stderr)
        self.assertIn('2 required artifact paths are missing', result.stderr)
        self.assertFalse((self.root / 'calls.jsonl').exists())

    def test_missing_clean_artifact_fails_before_training(self):
        self.env['MOTUS_WINDOWS'] = str(self.root / 'missing-windows.json')
        result = self.run_script()
        self.assertEqual(result.returncode, 2)
        self.assertIn('Missing MOTUS_WINDOWS', result.stderr)
        self.assertFalse((self.root / 'calls.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
