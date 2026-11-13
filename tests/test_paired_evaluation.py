"""Independent outcome/held-out-suite checks; no model weights or simulator."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

path = Path(__file__).resolve().parents[1]/'inference/robotwin/evaluate_pair.py'
spec = importlib.util.spec_from_file_location('paired_evaluation', path)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class PairedEvaluationTests(unittest.TestCase):
    def test_errors_and_timeouts_remain_in_denominator(self):
        rows = [dict(seed=i, status=status) for i,status in enumerate(
            ['success', 'success', 'timeout', 'invalid_prediction', 'reset_error'])]
        report = evaluation.summarize(rows, 5)
        self.assertEqual(report['success_rate'], .4)
        self.assertEqual(report['planned_scenes'], 5)
        self.assertEqual(report['outcomes']['reset_error'], 1)
        self.assertLess(report['wilson_95'][0], .4)
        self.assertGreater(report['wilson_95'][1], .4)

    def test_incomplete_suite_has_no_final_success_rate(self):
        report = evaluation.summarize([dict(seed=1, status='success')], 100)
        self.assertEqual(report['status'], 'incomplete')
        self.assertIsNone(report['success_rate'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            evaluation.summarize([dict(seed=1,status='success')]*2, 2)

    def test_test_seeds_exclude_validation_as_well_as_training(self):
        manifest = dict(identity='fixture', episodes=[dict(task_name='place_can_basket') for _ in range(500)])
        plan = dict(dataset_identity='fixture', task_name='place_can_basket', task_config='demo_randomized',
                    control_dt_s=.012, clock_provenance='synthetic test fixture, not a production clock',
                    cases=[dict(seed=1000+i,instruction='fixture instruction') for i in range(100)])
        evaluation.validate_plan(plan, manifest, list(range(500)))
        plan['cases'][0]['seed'] = 499
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            evaluation.validate_plan(plan, manifest, list(range(500)))
        plan['cases'][0]['seed'] = 1000
        plan['control_dt_s'] = None
        with self.assertRaisesRegex(ValueError, 'controller period'):
            evaluation.validate_plan(plan, manifest, list(range(500)))

    def test_native_controller_does_not_report_zero_simulated_duration(self):
        env = SimpleNamespace(take_action_cnt=0,step_lim=1,eval_success=False,get_obs=lambda:{})
        policy = SimpleNamespace(prediction_counts={})
        def advance(env, policy, observation):
            env.take_action_cnt = 1
            policy.last_controller_result = dict(action_commands=1,simulated_duration_s=None)
        result = evaluation.rollout(env,policy,SimpleNamespace(eval=advance))
        self.assertIsNone(result['simulated_duration_s'])

    def test_small_frozen_scene_suite_works_for_50_demo_pilot(self):
        manifest = dict(identity='fixture',episodes=[dict(task_name='place_can_basket') for _ in range(50)])
        plan = dict(dataset_identity='fixture',task_name='place_can_basket',task_config='demo_clean',
            control_dt_s=.012,clock_provenance='synthetic fixture',
            cases=[dict(seed=1000,instruction='fixture')])
        self.assertEqual(len(evaluation.validate_plan(plan,manifest,list(range(50)))),1)
        plan['cases'][0]['seed'] = 49
        with self.assertRaisesRegex(ValueError,'disjoint'):
            evaluation.validate_plan(plan,manifest,list(range(50)))

    def test_deployed_controller_success_timeout_and_invalidity(self):
        class Environment:
            take_action_cnt = 0
            step_lim = 32
            eval_success = False
            def get_obs(self): return {'current': True}
        def policy():
            return SimpleNamespace(prediction_counts=dict(invalid_predictions=0,repairs=0,fallbacks=0))
        def advance(env, model, observation):
            env.take_action_cnt += 16
            env.eval_success = env.take_action_cnt == 32
            model.last_controller_result = dict(action_commands=16,simulated_duration_s=.192)
        result = evaluation.rollout(Environment(), policy(), SimpleNamespace(eval=advance))
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['policy_calls'], 2)
        def never_succeed(env, model, observation):
            advance(env, model, observation)
            env.eval_success = False
        result = evaluation.rollout(Environment(), policy(), SimpleNamespace(eval=never_succeed))
        self.assertEqual(result['status'], 'timeout')
        self.assertEqual(result['controller_steps'], 32)
        def fail(env, model, observation):
            model.prediction_counts['invalid_predictions'] += 1
            raise ValueError('invalid spline support')
        result = evaluation.rollout(Environment(), policy(), SimpleNamespace(eval=fail))
        self.assertEqual(result['status'], 'invalid_prediction')
        self.assertEqual(result['policy_calls'], 1)
        self.assertEqual(result['counts']['fallbacks'], 0)


if __name__ == '__main__':
    unittest.main()
