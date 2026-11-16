"""CPU-only replay of saved predictions; never loads or changes model weights.

python -m data.robotwin2.redecode_predictions --checkpoint CHECKPOINT --output NEW_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from robotwin_spline import identity, read_cache

from .adaptive_spline import decode_local, extract_local
from .spline_diagnostics import score_spline_prediction


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.checkpoint / 'research_config.json'
    metadata = json.loads(metadata_path.read_text())
    contract = metadata['contract']
    if contract['representation'] != 'spline' or contract['identity'] != identity({k: v for k, v in contract.items() if k != 'identity'}):
        raise ValueError('Expected an intact spline checkpoint contract')
    generation = args.checkpoint.parent / 'generation'
    previous = json.loads((generation / 'results.json').read_text())
    if previous['contract_identity'] != contract['identity']:
        raise ValueError('Saved predictions have a different checkpoint contract')
    manifest_path = Path(metadata['config']['research']['window_manifest'])
    if sha256(manifest_path) != contract['window_manifest_sha256']:
        raise ValueError('Window manifest changed since checkpoint creation')
    windows = json.loads(manifest_path.read_text())['windows']
    by_id = {identity({k: v for k, v in w.items() if k != 'identity'}): w for w in windows}
    records, decoded = [], []
    with np.load(generation / 'predictions.npz', allow_pickle=False) as saved:
        raw, targets = saved['raw_parameters'], saved['target_commands']
        ids, seeds = saved['window_ids'].tolist(), saved['seeds'].tolist()
        expected = metadata['config']['research']['diagnostic']
        if seeds != expected['seeds'] or ids != [r['window_identity'] for r in previous['records'][::len(seeds)]]:
            raise ValueError('Saved generation windows/seeds differ from original results')
        if raw.shape != (len(ids), len(seeds), 36, 15):
            raise ValueError('Unexpected saved prediction shape')
        for i, window_id in enumerate(ids):
            window = by_id[window_id]
            actions, spline, cache_metadata = read_cache(window['cache_path'])
            if cache_metadata['normalization'] != contract['normalization']:
                raise ValueError('Cache normalization differs from checkpoint')
            obs = window['observation_index']
            prefix = contract['executed_prefix']
            times = np.minimum(obs + prefix['source_stride'] * np.arange(1, prefix['command_count'] + 1), len(actions) - 1)
            packet = extract_local(spline, obs, contract['knot_scaling'])
            target, _ = decode_local(packet['parameters'], times, obs, contract['knot_scaling'], contract['normalization'])
            np.testing.assert_allclose(target, targets[i], rtol=0, atol=2e-5)
            samples = []
            for j, seed in enumerate(seeds):
                commands, record = score_spline_prediction(raw[i, j], times, obs,
                    contract['knot_scaling'], contract['normalization'], targets[i])
                records.append(dict(record, window_identity=window_id, seed=seed,
                                    observation_index=obs, requested_timestamps=times.tolist()))
                samples.append(commands)
            decoded.append(samples)
    summary = dict(predictions=len(records),
        raw_valid=sum(r['raw_full_prefix_valid'] for r in records),
        repaired_spline_valid=sum(r['repaired_spline_valid'] for r in records),
        repaired_full_prefix_valid=sum(r['full_prefix_valid'] for r in records),
        covered_prefix_steps=sum(r['covered_steps'] for r in records),
        supported_steps=sum(r['supported_steps'] for r in records),
        requested_steps=sum(r['requested_steps'] for r in records))
    report = dict(checkpoint=str(args.checkpoint.resolve()), contract_identity=contract['identity'],
        source_predictions=str((generation / 'predictions.npz').resolve()),
        source_predictions_sha256=sha256(generation / 'predictions.npz'),
        checkpoint_metadata_sha256=sha256(metadata_path), window_manifest_sha256=sha256(manifest_path),
        validation='CPU re-decoding of original checkpoint generation; no model regeneration or retraining',
        seeds=seeds, window_ids=ids, summary=summary, records=records)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    np.savez_compressed(args.output / 'decoded.npz', decoded_commands=decoded, window_ids=ids, seeds=seeds)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
