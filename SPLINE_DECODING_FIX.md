# Method C decoding safeguard fix — 2026-11-16

SplineWAM [Appendix D, “Decoding safeguards,” page 18](https://arxiv.org/pdf/2609.39873#page=18) specifies running maximum, supported-domain evaluation, and trimming surplus endpoint repetitions. The existing adapter explicitly disabled repair; neither active Method C config exposed a repair switch. The shared strict evaluator already handled finiteness, cubic multiplicity, nondegenerate domains, float32 boundary roundoff and coverage correctly, so it is retained.

Absolute observation-relative knots, affine scaling, coefficient values/order, datasets and checkpoints are unchanged. Running maximum changes only knot coordinates. Endpoint trims retain coefficients `start:end-4`; interior multiplicity above four still fails. Evaluation uses the existing domain `[u[3], u[-4]]`, without extrapolation, retiming, sorting, positive increments, or replacement commands.

Offline generation and CPU replay call the same `decode_local` adapter recorded in the checkpoint deployment contract; repair defaults to enabled. Disabling repair is used only to measure raw validity. The legacy dense RoboTwin wrapper does not load research bundles or decode C; adding such a deployment loader is outside this focused fix. No deployed GPU/simulator rollout is claimed.

## Original artifacts

- Checkpoint: `outputs/ac_pilot_7643090/C/robotwin_pilot_C/pilot-seed-0/checkpoint_step_1000` (completed 1,000-update pilot).
- Predictions: the adjacent `generation/predictions.npz`; original four real training windows and seeds 0, 1, 2, 3.
- CPU replay: `outputs/spline-decoding-fix-20261116/results.json` and `decoded.npz` (fresh directory).
- Contract: `657340f22afc517b0ec9e6103a5997ff24e3792b24a24c991bdaf3b0614d1a3a`.
- Prediction SHA-256: `c37cf3a00e11dc91d68500d98bf55d9f7b31fa1dfd0fc905fcd8c8425393da4e`.
- Metadata SHA-256: `b2ea1d6feff4174b8bb3c68a9d80f97434d5a6bd47723c25845671cb5818c7a0`.
- Manifest SHA-256: `986298c3ee74a833dd2c14b7548d9661a8c5546aa621caf178c9978159e16115`.

Replay verifies checkpoint/prediction contract identity, manifest hash, window identities, seeds, cache normalization and original decoded targets. It reuses saved model outputs exactly; it does not regenerate predictions or load/train the model. CPU numerical changes need no GPU; any future GPU generation must use the existing Slurm allocation and `scripts/cluster/environment.sh --gpu` workflow.

## Results

Raw full-prefix validity: **0/16**. Repaired structural validity: **16/16**. Repaired full-prefix validity: **12/16 (75%)**. Execution prefix coverage: **192/256 commands (75%)**. Individually supported timestamps: **244/256**; those later timestamps cannot fill a missing first command. All four invalid prefixes have zero executable leading commands. No invalid prefix is returned as a complete command chunk.

Maximum correction: **35.6575 source steps**. Mean absolute correction across all 576 predicted knots: **1.14329 source steps**. All 16 predictions require repair. The table reports per-prediction maximum and mean correction before endpoint trimming.

| Window | Seed | Raw valid | Repaired prefix valid | Prefix / supported steps | Repair max / mean (steps) | Arm RMSE (rad) | Gripper RMSE (fraction) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | no | no | 0 / 14 of 16 | 7.5905 / 0.5783 | — | — |
| 0 | 1 | no | yes | 16 / 16 of 16 | 8.5832 / 1.0138 | 0.166980 | 0.031861 |
| 0 | 2 | no | no | 0 / 11 of 16 | 35.6575 / 3.4958 | — | — |
| 0 | 3 | no | yes | 16 / 16 of 16 | 7.8913 / 0.6958 | 0.081243 | 0.029707 |
| 1 | 0 | no | no | 0 / 15 of 16 | 4.4196 / 0.3961 | — | — |
| 1 | 1 | no | yes | 16 / 16 of 16 | 8.3220 / 1.1092 | 0.054638 | 0.042723 |
| 1 | 2 | no | no | 0 / 12 of 16 | 31.9130 / 3.0383 | — | — |
| 1 | 3 | no | yes | 16 / 16 of 16 | 7.9659 / 0.7172 | 0.102572 | 0.081410 |
| 2 | 0 | no | yes | 16 / 16 of 16 | 2.8611 / 0.1387 | 0.075375 | 0.049934 |
| 2 | 1 | no | yes | 16 / 16 of 16 | 7.7065 / 1.0704 | 0.045171 | 0.053862 |
| 2 | 2 | no | yes | 16 / 16 of 16 | 23.2074 / 2.1928 | 0.129311 | 0.035210 |
| 2 | 3 | no | yes | 16 / 16 of 16 | 7.1825 / 0.7142 | 0.109134 | 0.045710 |
| 3 | 0 | no | yes | 16 / 16 of 16 | 3.2248 / 0.1324 | 0.050923 | 0.050679 |
| 3 | 1 | no | yes | 16 / 16 of 16 | 8.0009 / 0.9224 | 0.047158 | 0.139758 |
| 3 | 2 | no | yes | 16 / 16 of 16 | 13.0978 / 1.4376 | 0.048796 | 0.090405 |
| 3 | 3 | no | yes | 16 / 16 of 16 | 6.7721 / 0.6398 | 0.089735 | 0.096233 |

Across the 12 complete valid prefixes, pooled arm RMSE is **0.0911012 rad** and gripper RMSE **0.0698859 fraction units**, against the original decoded target splines (per-channel errors also appear in JSON). Unsupported predictions have no full-prefix error metric. JSON separately reports errors on any supported timestamps and any covered execution prefix.

The rejected (window, seed) cases are (0,0), (0,2), (1,0), (1,2). Their repaired left domain boundaries are respectively **7.9711, 17.0286, 4.3152, 13.0743** source steps, beyond the first requested step **3**. Running maximum cannot restore missing early support. There is no evidence here of task success; large repairs and remaining coverage failures mean the checkpoint is not usable for every evaluated chunk.

## Reproduction and regression coverage

```bash
bash scripts/cluster/environment.sh --cpu python -m data.robotwin2.redecode_predictions \
  --checkpoint outputs/ac_pilot_7643090/C/robotwin_pilot_C/pilot-seed-0/checkpoint_step_1000 \
  --output outputs/spline-decoding-replay-NEW
bash scripts/cluster/environment.sh --cpu python -m unittest discover -s data/robotwin2 -p 'test_*.py' -v
bash scripts/cluster/environment.sh --cpu python -m unittest discover -s tests -p 'test_spline_integration.py' -v
```

**Passed:** 24 spline tests, 3 spline integration tests, and 5 existing deployment contract tests.

Coverage includes running maximum versus sorting, input/coefficient preservation, affine absolute times and negative support, leading/trailing endpoint alignment, excess interior multiplicity, degenerate domains, nonfinite packed values/overflow, unsupported coverage, and separate raw/repaired validity and errors. Existing packed-target/global-spline round trips remain covered.
