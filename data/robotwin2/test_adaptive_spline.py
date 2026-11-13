"""Lightweight numerical and converter contract checks, using synthetic actions."""
import contextlib
import io
import json
from pathlib import Path
import unittest
import uuid

import numpy as np
from scipy.interpolate import BSpline

from adaptive_spline import decode, fit_episode, normalize, read_cache, range_diagnostics
from convert_episode_spline import main
from adaptive_spline import (extract_local, decode_local, decode_episode_grid,
                            select_future_frames, fit_knot_time_scaling)


@contextlib.contextmanager
def fixture_directory():
    # Keep tiny fixtures in ignored outputs/, also on restricted Windows hosts
    # where tempfile's private-directory permissions cannot be reopened.
    root = Path(__file__).resolve().parents[2] / "outputs" / "spline-tests" / uuid.uuid4().hex
    root.mkdir(parents=True)
    yield root


class SplineChecks(unittest.TestCase):
    def test_cubic_initialization_and_independent_solver(self):
        x = np.linspace(0, 1, 20)
        y = np.column_stack([x**3, x**2, np.ones(20)])
        spline, d = fit_episode(y)
        self.assertTrue(d["success"])
        self.assertEqual(len(spline.t), 8)
        basis = BSpline.design_matrix(np.arange(20), spline.t, 3).toarray()
        coeff, _, rank, _ = np.linalg.lstsq(basis, y, rcond=None)
        self.assertEqual(rank, 4)
        np.testing.assert_allclose(coeff, spline.c, atol=1e-12)

    def test_switch_refinement_and_cap(self):
        y = np.column_stack([np.sin(np.arange(60)), np.arange(60) >= 30]).astype(float)
        spline, d = fit_episode(y)
        self.assertTrue(d["success"])
        self.assertLessEqual(d["max_abs_normalized_residual"], 0.01)
        self.assertLessEqual(len(spline.c), len(y))
        _, capped = fit_episode(y, max_control_points=4)
        self.assertFalse(capped["success"])
        self.assertEqual(capped["reason"], "control_point_cap")

    def test_constants_no_clipping_and_rejection(self):
        norm = dict(low=[0, 7], high=[2, 7])
        a = np.array([[3., 7.]])
        self.assertEqual(normalize(a, norm)[0, 0], 2)
        np.testing.assert_allclose(decode(normalize(a, norm), norm), a)
        with self.assertRaises(ValueError):
            normalize(np.array([[3., 8.]]), norm)
        with self.assertRaises(ValueError):
            fit_episode(np.zeros((3, 14)))

    def test_curve_overshoot_extrema(self):
        # A cubic with an interior maximum missed by the sparse action grid.
        x = np.arange(4.)
        y = (-(x - 1.5)**2)[:, None]
        spline, _ = fit_episode(y)
        norm = dict(low=[0], high=[2])  # decode(y)=y+1
        ranges = range_diagnostics(spline, y + 1, norm)
        self.assertAlmostEqual(ranges["overshoot_above"][0], 0.25)

    def test_converter_split_roundtrip_and_normalization_reuse(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            episodes = root / "episodes"
            episodes.mkdir()
            # Split determined by seed, before values; validation is an outlier.
            val_index = np.random.default_rng(0).permutation(4)[0]
            for i in range(4):
                a = np.tile(np.linspace(0, 1, 12)[:, None], (1, 14)) + (100 if i == val_index else i)
                np.save(episodes / f"{i}.npy", a)
            out = root / "fit.npz"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--episode-dir", str(episodes), "--output", str(out)]), 0)
            actions, spline, meta = read_cache(out)
            self.assertLess(max(meta["normalization"]["high"]), 100)
            np.testing.assert_allclose(decode(spline(np.arange(len(actions))), meta["normalization"]), actions, atol=1e-10)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--episode-dir", str(episodes), "--episode", str(episodes / f"{val_index}.npy"),
                    "--normalization", str(out.with_suffix(".normalization.json")), "--output", str(root / "validation.npz")]), 0)
            _, _, vm = read_cache(root / "validation.npz")
            self.assertEqual(vm["normalization"]["identity"], meta["normalization"]["identity"])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--episode-dir", str(episodes), "--output", str(out)])

    def test_single_episode_requires_diagnostic_opt_in(self):
        with fixture_directory() as tmp:
            root = Path(tmp)
            np.save(root / "single.npy", np.zeros((10, 14)))
            args = ["--episode-dir", str(root), "--output", str(root / "fit.npz")]
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args + ["--diagnostic-only"]), 0)
            _, _, meta = read_cache(root / "fit.npz")
            self.assertEqual(meta["normalization"]["scope"], "diagnostic-only")

    def test_capped_converter_saves_failed_artifact(self):
        with fixture_directory() as root:
            a = np.tile(np.sin(np.arange(60))[:, None], (1, 14))
            np.save(root / "switch.npy", a)
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["--episode-dir", str(root), "--output", str(root / "failed.npz"),
                             "--diagnostic-only", "--max-control-points", "4"])
            self.assertEqual(code, 2)
            _, _, meta = read_cache(root / "failed.npz")
            self.assertFalse(meta["diagnostics"]["success"])
            self.assertEqual(meta["diagnostics"]["reason"], "control_point_cap")


class LocalTargetChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.linspace(0, 12, 180)
        cls.y = np.column_stack([np.sin((j + 1) * x) for j in range(14)])
        cls.y[:, 6] = (x >= 4).astype(float)
        cls.spline, _ = fit_episode(cls.y)
        cls.norm = dict(low=[-1.] * 14, high=[1.] * 14)
        cls.scaling = dict(scope="diagnostic-only", units="source_action_step", offset=17., scale=100.)

    def test_roundtrip_entire_domain_and_boundaries_float32(self):
        worst = 0.
        for obs in [0., .25, 1., 55.5, 130.2, 178., 179.]:
            sample = extract_local(self.spline, obs, self.scaling)
            self.assertEqual(sample["parameters"].shape, (36, 15))
            self.assertEqual(sample["parameters"].dtype, np.float32)
            lo, hi = np.array(sample["valid_interval_relative"]) + obs
            times = np.unique(np.r_[np.linspace(lo, hi, 401), obs])
            result, report = decode_local(sample["parameters"], times, obs, self.scaling, self.norm)
            worst = max(worst, float(np.abs(result - self.spline(times)).max()))
            self.assertFalse(report["extrapolated"])
            self.assertFalse(report["repaired_knots"])
        self.assertLess(worst, 2e-5)
        sample = extract_local(self.spline, 1.25, self.scaling)
        knots = sample["parameters"][:, 0] * self.scaling["scale"] + self.scaling["offset"]
        self.assertLess(knots[0], 0)  # retain past support, not clipped to observation

    def test_whole_controller_grid_uses_same_decoder(self):
        times = np.linspace(0, 179, 1200)
        actual, reports = decode_episode_grid(self.spline, times, self.scaling, self.norm)
        np.testing.assert_allclose(actual, self.spline(times), atol=2e-5)
        self.assertEqual(len(reports), len(times))

    def test_actual_packed_npz_serialization(self):
        sample = extract_local(self.spline, 63.4, self.scaling)
        with fixture_directory() as root:
            np.savez_compressed(root / "target.npz", parameters=sample.pop("parameters"), metadata=json.dumps(sample))
            with np.load(root / "target.npz", allow_pickle=False) as f:
                m = json.loads(str(f["metadata"]))
                result, _ = decode_local(f["parameters"], [63.4, 64.], m["observation_time"], m["scaling"], self.norm)
            np.testing.assert_allclose(result, self.spline([63.4, 64.]), atol=2e-5)

    def test_reference_chunk_packing_equivalence(self):
        # Execute the two read-only pure reference functions without importing
        # its torch/diffusion-policy/replay-buffer dependency tree.
        import ast
        import typing
        from types import SimpleNamespace
        reference = Path(__file__).resolve().parents[3] / "bspline-policy/bspline_policy/bspline_policy/common/bspline_action.py"
        tree = ast.parse(reference.read_text())
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
                 ("extract_unique_knots", "chunk_bspline_trajectory")]
        namespace = dict(np=np, Optional=typing.Optional, ScipyBSplineCompression=object)
        exec(compile(ast.Module(body=funcs, type_ignores=[]), str(reference), "exec"), namespace)
        chunks = namespace["chunk_bspline_trajectory"](SimpleNamespace(spline=self.spline, degree=3), chunk_size=30, stride=1)
        for chunk in [chunks[0], chunks[len(chunks)//2], chunks[-1]]:
            observation = float(chunk["t"][3])
            sample = extract_local(self.spline, observation, dict(self.scaling, offset=0., scale=1.))
            expected = np.column_stack((chunk["t"] - observation, chunk["c"])).astype(np.float32)
            np.testing.assert_array_equal(sample["parameters"], expected)

    def test_invalid_predictions_and_coverage(self):
        sample = extract_local(self.spline, 0., self.scaling)
        bad = sample["parameters"].copy()
        bad[5, 0] = bad[4, 0] - 1
        with self.assertRaisesRegex(ValueError, "decreasing"):
            decode_local(bad, [0.], 0., self.scaling, self.norm)
        bad = sample["parameters"].copy()
        bad[1, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            decode_local(bad, [0.], 0., self.scaling, self.norm)
        with self.assertRaisesRegex(ValueError, "coverage"):
            decode_local(sample["parameters"], [179.], 0., self.scaling, self.norm)
        with self.assertRaises(ValueError):
            decode_local(np.zeros((36, 15)), [0.], 0., self.scaling, self.norm)
        with self.assertRaises(ValueError):
            extract_local(self.spline, 180., self.scaling)
        with self.assertRaises(ValueError):
            decode_local(sample["parameters"], [1., 0.], 0., self.scaling, self.norm)

    def test_tail_padding_trim_and_continuous_gripper(self):
        sample = extract_local(self.spline, 179., self.scaling)
        result, report = decode_local(sample["parameters"], [179.], 179., self.scaling, self.norm)
        self.assertEqual(sample["effective_control_points"], 4)
        self.assertEqual(report["trimmed_tail_knots"], 28)
        self.assertGreater(result[0, 13], -1)
        self.assertNotIn(float(result[0, 13]), (0., 1.))
        np.testing.assert_allclose(result, self.spline([179.]), atol=2e-5)

    def test_deterministic_frame_clock_and_vae_padding(self):
        sample = extract_local(self.spline, 0., self.scaling)
        aclock, cclock = np.arange(180) * .1, np.arange(359) * .05
        selection = select_future_frames(sample, aclock, cclock)
        self.assertEqual(selection, select_future_frames(sample, aclock, cclock))
        self.assertEqual([r["knot_row"] for r in selection["frames"]], list(range(4, 33, 4)))
        for frame in selection["frames"]:
            self.assertAlmostEqual(cclock[frame["camera_index"]], frame["target_timestamp"], places=5)
        end = extract_local(self.spline, 179., self.scaling)
        padded = select_future_frames(end, aclock, cclock)
        self.assertFalse(padded["unchanged_motus_training_compatible"])
        self.assertEqual(padded["conservative_latent_valid"], [True, False, False])
        self.assertEqual(padded["proposed_latent_loss_mask"], [False, False, False])
        self.assertTrue(all(r["repeated_frame"] for r in padded["frames"]))
        with self.assertRaises(ValueError):
            select_future_frames(sample, aclock, cclock, count=6)

    def test_knot_scaling_excludes_validation_and_requires_full_training_split(self):
        with fixture_directory() as root:
            episodes = root / "episodes"
            episodes.mkdir()
            for i in range(4):
                np.save(episodes / f"{i}.npy", np.tile(np.linspace(0, i + 1, 12)[:, None], (1, 14)))
            first = root / "first.npz"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--episode-dir", str(episodes), "--output", str(first)])
            _, _, meta = read_cache(first)
            normfile = first.with_suffix(".normalization.json")
            caches = []
            for i, source in enumerate(meta["normalization"]["training_sources"]):
                out = root / f"train{i}.npz"
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["--episode-dir", str(episodes), "--episode", source["path"], "--output", str(out), "--normalization", str(normfile)])
                caches.append(out)
            scaling = fit_knot_time_scaling(caches)
            self.assertEqual(scaling["scope"], "training-only")
            self.assertEqual(scaling["normalization_identity"], meta["normalization"]["identity"])
            with self.assertRaisesRegex(ValueError, "exactly all"):
                fit_knot_time_scaling(caches[:1])
            validation = root / "validation.npz"
            with contextlib.redirect_stdout(io.StringIO()):
                main(["--episode-dir", str(episodes), "--episode", meta["normalization"]["validation_paths"][0],
                      "--output", str(validation), "--normalization", str(normfile)])
            with self.assertRaisesRegex(ValueError, "Validation"):
                fit_knot_time_scaling(caches + [validation])


class ReplayHookChecks(unittest.TestCase):
    def test_existing_step_schedule_velocity_and_gripper_eps_are_retained(self):
        # A simulator-free hook contract test; this is NOT simulator replay.
        from replay_episode_spline import run_trial
        class Scene:
            def step(self):
                pass
            def get_timestep(self):
                return 1 / 250
            def get_all_actors(self):
                return []
        class Robot:
            def __init__(self):
                self.state = np.zeros(14)
                self.velocities, self.eps = [], []
            def set_arm_joints(self, position, velocity, tag):
                self.state[:6] = position
                self.velocities.append(list(velocity))
            def set_gripper(self, value, tag, gripper_eps=.1):
                self.state[6] = value
                self.eps.append(gripper_eps)
            def get_left_arm_jointState(self):
                return self.state[:7].tolist()
            def get_right_arm_jointState(self):
                return self.state[7:].tolist()
            get_left_arm_real_jointState = get_left_arm_jointState
            get_right_arm_real_jointState = get_right_arm_jointState
        class Env:
            def __init__(self):
                self.scene, self.robot = Scene(), Robot()
                self.closed = False
            def setup_demo(self, **kwargs):
                pass
            def set_path_lst(self, config):
                pass
            def load_tran_data(self, episode):
                return {}
            def get_arm_pose(self, tag):
                return [0, 0, 0, 1, 0, 0, 0]
            def play_once(self):
                self._take_picture()
                for _ in range(3):
                    self.robot.set_arm_joints(np.ones(6), np.full(6, 42), "left")
                    self.robot.set_gripper(.75, "left", .07)
                    self.scene.step()
                    self._take_picture()
            def check_success(self):
                return True
            def close_env(self):
                self.closed = True
        env = Env()
        report = run_trial(lambda task: env, {"task_name": "fake"}, 0, 0, commands=np.full((4, 14), .25))
        self.assertEqual(report["steps"], 3)
        self.assertEqual(report["picture_steps"].tolist(), [0, 1, 2, 3])
        self.assertEqual(env.robot.velocities, [[42] * 6] * 3)
        self.assertEqual(env.robot.eps, [.07] * 3)
        self.assertTrue(env.closed)
        self.assertTrue(np.all(report["commanded"][:, :7] == .25))


if __name__ == "__main__":
    unittest.main()
