"""Regression coverage for strict scoring and simulator timing aggregation."""
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from experiments.ac_spline_pilot.offline import error_totals, score_prediction, summarize
from experiments.ac_spline_pilot.simulator import summarize_episodes


class EvaluationTests(unittest.TestCase):
    def test_partial_spline_support_keeps_original_grid_and_denominator(self):
        parameters = np.zeros((36, 15))
        parameters[:, 0] = np.r_[np.full(4, 2.), np.linspace(2.1, 7.9, 28), np.full(4, 8.)]
        contract = dict(representation='spline',
            knot_scaling=dict(scope='diagnostic-only', units='source_action_step', offset=0., scale=1.),
            normalization=dict(low=[-1.] * 14, high=[1.] * 14))
        rows = np.array([10, 13, 16, 19, 22])
        commands, mask, diagnostics = score_prediction(parameters, rows, 10, contract)
        np.testing.assert_array_equal(mask, [False, True, True, False, False])
        self.assertEqual(commands.shape, (2, 14))
        self.assertTrue(diagnostics['decoder_success'])
        self.assertFalse(diagnostics['coverage_valid'])

        class FK:
            def poses(self, actions):
                poses = np.zeros((len(actions), 2, 7))
                poses[..., 0] = actions[:, 0, None]
                quaternions = Rotation.from_rotvec(actions[:, :3]).as_quat()
                poses[..., 3:] = quaternions[:, None, [3, 0, 1, 2]]
                return poses

        target = commands.copy()
        target[:, 0] += .25
        errors = error_totals(FK(), commands, target)
        result = summarize([dict(source_rows=rows.tolist(), supported=errors)])
        self.assertEqual(result['full_window_coverage'], 0.)
        self.assertEqual(result['timestamp_coverage'], .4)
        self.assertIsNone(result['full_window_errors']['ee_position_rmse_m'])
        self.assertAlmostEqual(result['supported_errors']['ee_position_rmse_m'], .25)
        self.assertAlmostEqual(result['supported_errors']['ee_orientation_mean_rad'], .25)

    def test_latency_denominators_and_official_success(self):
        records = [
            dict(success=True, official_success=True, simulated_seconds=.8, episode_wall_seconds=10.,
                 policy_calls=1, commands_per_call=[2], calls=[
                     dict(inference_wall_seconds=.3, total_wall_seconds=.5)]),
            dict(success=False, official_success=False, simulated_seconds=0., episode_wall_seconds=4.,
                 policy_calls=1, commands_per_call=[0], failure_reason='decoder error', calls=[
                     dict(inference_wall_seconds=.4, decode_wall_seconds=.01, total_wall_seconds=.6,
                          decoded_valid=False, spline_diagnostics=dict(coverage_valid=False))]),
        ]
        result = summarize_episodes(records)
        self.assertEqual(result['success_rate'], .5)
        self.assertEqual(result['average_commands_per_call'], 1.)
        self.assertAlmostEqual(result['average_simulated_seconds'], .4)
        self.assertAlmostEqual(result['average_inference_wall_seconds'], .35)
        self.assertAlmostEqual(result['average_decode_wall_seconds'], .01)
        self.assertEqual(result['failure_categories'], {'spline_decode_error': 1})
        self.assertEqual(result['spline_diagnostics']['decoder_failures'], 1)
        self.assertEqual(result['spline_diagnostics']['temporal_coverage_failures'], 1)


if __name__ == '__main__':
    unittest.main()
