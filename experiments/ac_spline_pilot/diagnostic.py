"""Evaluation-only tiny-data diagnostic called by the existing MOTUS trainer.

No training loop here. Generation uses only current image, state and language;
future frames and spline parameters are used exclusively for scoring/plots.
"""
import json
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def generate_diagnostic(model, dataloader, config, output, reference_checkpoint=None):
    from data.dataset import collate_fn
    from robotwin_spline import CHANNELS
    from data.robotwin2.adaptive_spline import decode_local
    from data.robotwin2.spline_diagnostics import score_spline_prediction
    from train.sample import inference_sample
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = dataloader.dataset
    seeds = list(config.research.diagnostic.seeds)
    count = min(int(config.research.diagnostic.windows), len(dataset))
    contract = model.research_contract
    raw, decoded, targets, records, sample_ids = [], [], [], [], []
    # Fixed indices, not the shuffled first training batch. Language selection
    # is fixed by each window's explicit language_index (default zero).
    for index in range(count):
        batch = collate_fn([dataset[index]])
        info = batch['research_metadata'][0]
        sample_ids.append(info['window_identity'])
        obs = info['observation_index']
        times = np.asarray(info['dense_indices'], dtype=float)
        target = batch['action_sequence'][0].float().numpy()
        if contract['representation'] == 'spline':
            target_commands, _ = decode_local(target, times, obs, contract['knot_scaling'], contract['normalization'])
        else:
            target_commands = target
        targets.append(target_commands)
        sample_predictions, sample_decoded = [], []
        fig, axes = plt.subplots(7, 2, figsize=(16, 18))
        axes = axes.ravel()
        for j in range(14):
            axes[j].plot(times, target_commands[:, j], color='black', lw=1, label='target')
            axes[j].set_title(CHANNELS[j] + (' [fraction]' if j in (6,13) else ' [rad]'))
            axes[j].set_xlabel('source action-step index')
            if j in (6,13):
                axes[j].set_facecolor('#f2f2f2')
        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # inference_sample never reads future frames; only the current
            # observation, proprioception, T5 context and current-frame VLM input.
            _, prediction = inference_sample(model, batch, config)
            parameters = prediction[0].float().cpu().numpy()
            sample_predictions.append(parameters)
            record = dict(window_identity=info['window_identity'], seed=seed, full_prefix_valid=False,
                          covered_steps=0, decoded_rmse_per_channel=None)
            commands = np.full((len(times), 14), np.nan)
            try:
                if contract['representation'] == 'spline':
                    commands, validity = score_spline_prediction(
                        parameters, times, obs, contract['knot_scaling'],
                        contract['normalization'], target_commands)
                    record.update(validity)
                    if not record['full_prefix_valid']:
                        raise ValueError(record['invalid_reason'])
                else:
                    if not np.isfinite(parameters).all():
                        raise ValueError('Nonfinite dense predictions')
                    commands = parameters
                record['full_prefix_valid'] = True
                record['covered_steps'] = len(times)
                record['decoded_rmse_per_channel'] = np.sqrt(np.mean((commands - target_commands)**2, axis=0)).tolist()
                for j in range(14):
                    axes[j].plot(times, commands[:, j], alpha=.6, lw=.8, label=f'seed {seed}')
            except ValueError as exc:
                record['invalid_reason'] = str(exc)
                fig.text(.03, .02 + .018 * seeds.index(seed), f'seed {seed}: {exc}', fontsize=8, color='red')
            sample_decoded.append(commands)
            records.append(record)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{contract['representation']} | observation {obs} | {info['window_identity'][:12]}")
        fig.tight_layout(rect=(0,.11,1,.97))
        fig.savefig(output / f'window-{index}.png', dpi=120)
        plt.close(fig)
        raw.append(sample_predictions)
        decoded.append(sample_decoded)
    raw, decoded = np.asarray(raw), np.asarray(decoded)
    np.savez_compressed(output / 'predictions.npz', raw_parameters=raw, decoded_commands=decoded,
                        target_commands=np.asarray(targets), seeds=seeds, window_ids=sample_ids)
    report = dict(contract_identity=contract['identity'], model='full MOTUS', real_windows=count,
                  representation=contract['representation'], seeds=seeds, records=records,
                  valid_fraction=float(np.mean([r['full_prefix_valid'] for r in records])),
                  raw_valid_fraction=float(np.mean([r.get('raw_full_prefix_valid', r['full_prefix_valid']) for r in records])),
                  target_rows=model.config.action_chunk_size,
                  action_branch_tokens=model.config.action_chunk_size + 1 + model.action_expert.config.num_registers,
                  reload_check='not_requested')
    if reference_checkpoint:
        previous = Path(reference_checkpoint).parent / 'generation' / 'predictions.npz'
        with np.load(previous, allow_pickle=False) as original:
            if original['window_ids'].tolist() != sample_ids or original['seeds'].tolist() != seeds:
                raise ValueError('Fresh reload compared different windows/noise seeds')
            difference = float(np.max(np.abs(raw - original['raw_parameters'])))
        report['reload_max_abs_parameter_difference'] = difference
        report['reload_check'] = 'passed' if difference <= 1e-5 else 'failed'
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    model.train()
    if report['reload_check'] == 'failed':
        raise ValueError('Fresh-process generation changed after checkpoint reload')
    return report
