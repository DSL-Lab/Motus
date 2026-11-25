"""Regression checks for Appendix D prediction safeguards and prefix scoring."""
import unittest

import numpy as np
from scipy.interpolate import BSpline

from data.robotwin2.adaptive_spline import decode_execution, decode_local, extract_local, prepare_local
from data.robotwin2.spline_diagnostics import score_spline_prediction


class PredictionSafeguards(unittest.TestCase):
    def setUp(self):
        self.scaling = dict(scope='diagnostic-only', units='source_action_step', offset=0., scale=1.)
        self.norm = dict(low=[-1.] * 14, high=[1.] * 14)
        self.packet = np.column_stack((np.r_[np.zeros(4), np.arange(1., 29.), np.full(4, 29.)],
                                       np.arange(36 * 14).reshape(36, 14) / 500.))

    def test_running_maximum_keeps_coefficient_order_and_input(self):
        self.packet[10, 0] = self.packet[9, 0] - 2
        original = self.packet.copy()
        knots, cp, report = prepare_local(self.packet, self.scaling)
        np.testing.assert_array_equal(knots, np.maximum.accumulate(original[:, 0]))
        np.testing.assert_array_equal(cp, original[:32, 1:])
        np.testing.assert_array_equal(self.packet, original)
        self.assertTrue(report['repaired_knots'])
        self.assertEqual(report['repaired_knot_count'], 1)
        self.assertEqual(report['knot_repair_max_source_steps'], 2.)
        self.assertAlmostEqual(report['knot_repair_mean_source_steps'], 2 / 36)
        times = np.arange(1., 29.)
        actual, report = decode_local(self.packet, times, 0., self.scaling, self.norm)
        np.testing.assert_allclose(actual, BSpline(knots, original[:32, 1:], 3, extrapolate=False)(times))
        self.assertTrue(report['repaired_knots'])
        self.assertFalse(report['extrapolated'])
        with self.assertRaisesRegex(ValueError, 'decreasing'):
            decode_local(self.packet, times, 0., self.scaling, self.norm, repair_knots=False)

    def test_absolute_affine_time_axis_and_past_support_are_preserved(self):
        self.packet[:, 0] -= 5.
        raw_knots = self.packet[:, 0].copy()
        self.packet[:, 0] = (raw_knots - 17.) / 100.
        scaling = dict(self.scaling, offset=17., scale=100.)
        times = np.arange(0., 24.)
        commands, report = decode_local(self.packet, times + 55., 55., scaling, self.norm)
        expected = BSpline(raw_knots, self.packet[:32, 1:], 3, extrapolate=False)(times)
        np.testing.assert_allclose(commands, expected, atol=1e-14)
        self.assertEqual(report['valid_interval_relative'], [-5., 24.])
        self.assertFalse(report['repaired_knots'])

    def test_both_endpoint_trims_preserve_coefficient_association(self):
        self.packet[:, 0] = np.r_[np.zeros(6), np.arange(1., 23.), np.full(8, 23.)]
        times = np.linspace(0., 23., 101)
        commands, report = decode_local(self.packet, times, 0., self.scaling, self.norm)
        expected = BSpline(self.packet[2:32, 0], self.packet[2:28, 1:], 3, extrapolate=False)(times)
        np.testing.assert_allclose(commands, expected)
        self.assertEqual(report['trimmed_head_knots'], 2)
        self.assertEqual(report['trimmed_tail_knots'], 4)
        self.assertEqual(report['effective_control_points'], 26)

    def test_uniform_and_nonuniform_windows_use_canonical_active_domain(self):
        for knots in (np.arange(36.) - 5., np.cumsum(np.linspace(.5, 2., 36)) - 5.):
            self.packet[:, 0] = knots
            domain = knots[[3, 32]]
            times = np.linspace(*domain, 51)
            commands, report = decode_local(self.packet, times, 0., self.scaling, self.norm)
            np.testing.assert_allclose(commands, BSpline(knots, self.packet[:32, 1:], 3)(times), atol=1e-14)
            self.assertEqual(report['valid_interval_relative'], domain.tolist())
            self.assertEqual(report['effective_control_points'], 32)

    def test_excess_at_active_boundary_and_repeated_nonclamped_right_endpoint(self):
        self.packet[:, 0] = np.r_[[-2., -1.], np.zeros(5), np.arange(1., 22.),
                                 np.full(6, 22.), [23., 24.]]
        knots, coefficients, report = prepare_local(self.packet, self.scaling)
        self.assertEqual(report['valid_interval_relative'], [0., 22.])
        self.assertEqual((report['trimmed_head_knots'], report['trimmed_tail_knots']), (3, 4))
        np.testing.assert_array_equal(coefficients, self.packet[3:28, 1:])
        values, _ = decode_local(self.packet, [0., 22.], 0., self.scaling, self.norm)
        np.testing.assert_allclose(values, coefficients[[0, -1]])
        self.packet[:, 0] = np.arange(36.)
        self.packet[31, 0] = 32.
        values, _ = decode_local(self.packet, [32.], 0., self.scaling, self.norm)
        expected = BSpline(self.packet[:, 0], self.packet[:32, 1:], 3)(np.nextafter(32., 0.))
        np.testing.assert_allclose(values[0], expected, atol=1e-14)

    def test_local_target_uses_observation_span_and_preserves_parent_curve(self):
        knots = np.r_[np.zeros(4), np.arange(2., 98., 2), np.full(4, 98.)]
        control = np.arange((len(knots) - 4) * 14).reshape(-1, 14) / 1000.
        parent = BSpline(knots, control, 3, extrapolate=False)
        scaling = dict(self.scaling, offset=27., scale=100.)
        for observation in (0, 13, 14, 93):
            packet = extract_local(parent, observation, scaling)
            span = np.searchsorted(knots, observation, side='right') - 1
            self.assertEqual(packet['global_start'], span - 3)
            lower, upper = packet['valid_interval_relative']
            self.assertLessEqual(lower, 0.)
            times = np.arange(observation, observation + upper + 1.)
            commands, report = decode_local(packet['parameters'], times, observation, scaling, self.norm,
                                            repair_knots=False)
            np.testing.assert_allclose(commands, parent(times), atol=1e-7)
            np.testing.assert_allclose(report['valid_interval_relative'], [lower, upper], atol=5e-6)

    def test_adaptive_execution_keeps_stride_and_reports_failed_first_command(self):
        self.packet[:, 0] = np.arange(36.) * 2 - 8
        self.packet[8, 0] = 5.
        diagnostics = {}
        prefix = dict(row_budget=12, source_stride=3)
        commands, report = decode_execution(self.packet, self.scaling, self.norm, prefix, diagnostics=diagnostics)
        self.assertEqual(commands.shape, (2, 14))
        self.assertEqual(report['valid_interval_relative'], [-2., 6.])
        self.assertEqual(diagnostics['requested_interval_relative'], [3., 6.])
        self.assertTrue(diagnostics['coverage_valid'])
        self.assertEqual(diagnostics['repaired_knot_count'], 1)
        self.assertAlmostEqual(diagnostics['repair_rate'], 1 / 36)
        self.assertAlmostEqual(diagnostics['repair_displacement'], 1 / 36)
        # Reproduce a backward-moving support knot that pushes U[3] late.
        self.packet[1:4, 0] = [4., -3., -2.]
        with self.assertRaisesRegex(ValueError, 'Insufficient coverage'):
            decode_execution(self.packet, self.scaling, self.norm, prefix, diagnostics=diagnostics)
        self.assertEqual(diagnostics['raw_knots'][3], -2.)
        self.assertEqual(diagnostics['repaired_knots_vector'][3], 4.)
        self.assertEqual(diagnostics['requested_interval_relative'][0], 3.)
        self.assertFalse(diagnostics['coverage_valid'])

    def test_invalid_interior_multiplicity_and_degenerate_domain_rejected(self):
        self.packet[10:15, 0] = self.packet[10, 0]
        with self.assertRaisesRegex(ValueError, 'multiplicity'):
            decode_local(self.packet, [3.], 0., self.scaling, self.norm)
        self.packet[:, 0] = 0.
        with self.assertRaisesRegex(ValueError, 'multiplicity|Degenerate'):
            decode_local(self.packet, [0.], 0., self.scaling, self.norm)

    def test_valid_internal_multiplicity_uses_right_piece_at_the_knot(self):
        self.packet[:, 0] = np.repeat([0., 1., 2.], [4, 4, 28])
        self.packet[:4, 1:] = .2
        self.packet[4:, 1:] = .8
        times = [0., .7, 1., 1.4, 2.]
        values, _ = decode_local(self.packet, times, 0., self.scaling, self.norm)
        np.testing.assert_allclose(values, np.repeat([[.2], [.2], [.8], [.8], [.8]], 14, axis=1))

    def test_running_maximum_does_not_repair_excess_interior_repetitions(self):
        self.packet[10:15, 0] = [-1., -2., -3., -4., -5.]
        with self.assertRaisesRegex(ValueError, 'multiplicity'):
            decode_local(self.packet, [3.], 0., self.scaling, self.norm)

    def test_nonfinite_parameters_scaling_and_queries_rejected(self):
        for row, col in [(1, 0), (35, 14)]:
            for value in [np.nan, np.inf, -np.inf]:
                bad = self.packet.copy()
                bad[row, col] = value
                with self.assertRaisesRegex(ValueError, 'finite'):
                    decode_local(bad, [3.], 0., self.scaling, self.norm)
        bad = self.packet.copy()
        bad[:, 0] = np.finfo(float).max
        with self.assertRaisesRegex(ValueError, 'Nonfinite knots'):
            decode_local(bad, [3.], 0., dict(self.scaling, scale=2.), self.norm)
        for times in [[np.nan], [4., 3.], []]:
            with self.assertRaises(ValueError):
                decode_local(self.packet, times, 0., self.scaling, self.norm)

    def test_prefix_scoring_does_not_extrapolate_or_retime(self):
        times = np.arange(3., 49., 3.)
        commands, record = score_spline_prediction(self.packet, times, 0., self.scaling, self.norm, np.zeros((16, 14)))
        self.assertFalse(record['raw_full_prefix_valid'])
        self.assertTrue(record['repaired_spline_valid'])
        self.assertFalse(record['full_prefix_valid'])
        self.assertEqual(record['covered_steps'], 9)
        self.assertEqual(record['supported_steps'], 9)
        self.assertEqual(record['execution_prefix_coverage'], 9 / 16)
        self.assertIsNone(record['decoded_rmse_per_channel'])
        self.assertIsNotNone(record['covered_prefix_rmse_per_channel'])
        self.assertTrue(np.isnan(commands[9:]).all())
        self.packet[:, 0] += 4
        commands, record = score_spline_prediction(self.packet, [3., 6., 9.], 0., self.scaling, self.norm, np.zeros((3, 14)))
        self.assertEqual(record['covered_steps'], 0)
        self.assertEqual(record['supported_steps'], 2)
        self.assertTrue(np.isnan(commands).all())

    def test_repaired_and_raw_validity_are_separate(self):
        self.packet[10, 0] = self.packet[9, 0] - 1
        times = np.arange(3., 28., 3.)
        expected = BSpline(np.maximum.accumulate(self.packet[:, 0]), self.packet[:32, 1:], 3)(times)
        commands, record = score_spline_prediction(self.packet, times, 0., self.scaling, self.norm, expected)
        self.assertFalse(record['raw_full_prefix_valid'])
        self.assertTrue(record['full_prefix_valid'])
        self.assertEqual(record['covered_steps'], len(times))
        np.testing.assert_allclose(record['decoded_rmse_per_channel'], 0., atol=1e-14)
        np.testing.assert_allclose(commands, expected)


if __name__ == '__main__':
    unittest.main()
