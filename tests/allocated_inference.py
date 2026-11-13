"""Exercise the existing deployment wrapper on saved tiny-data checkpoints.

One current demonstration image/state/instruction only. No simulator rollout,
future images or success claim; retain raw C predictions and every repair/failure.
"""
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from inference.robotwin.Motus.deploy_policy import MotusPolicy
from motus_actions.prepare import read_frames

if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
    raise RuntimeError('requires allocated GPU')
output=ROOT/'outputs'/f'wrapper-inference-{os.environ["SLURM_JOB_ID"]}'
output.mkdir(exist_ok=False)
results=[]
selected=sys.argv[1:] or ['A','C']
if any(r not in ('A','C') for r in selected): raise ValueError('expected A and/or C')
for representation,job in [(r, '7471646' if r=='A' else '7471647') for r in selected]:
    checkpoint=ROOT/'outputs'/f'tiny-{representation}-{job}'/'checkpoint_step_50'
    config_path=checkpoint/'resolved_config.yaml'
    config=yaml.safe_load(config_path.read_text())
    manifest=json.loads(Path(config['dataset']['episode_manifest']).read_text())
    episode=next(e for e in manifest['episodes'] if e['id'] in manifest['split']['validation'])
    with np.load(episode['cache_path']) as cache:
        observation=int(cache['observations'][len(cache['observations'])//2])
        state=cache['states'][observation].copy()
    image=read_frames(episode['hdf5_path'],[observation])[0]
    policy=MotusPolicy(str(checkpoint),str(config_path),config['model']['wan']['config_path'],
        config['model']['vlm']['checkpoint_path'],device='cuda',log_dir=str(output/representation))
    policy.set_instruction(episode['instruction'])
    observation_inputs={'image':image,'joint_action':{'vector':state}}
    torch.cuda.manual_seed_all(13)
    torch.cuda.synchronize(); start=time.perf_counter()
    policy.update_obs(observation_inputs)
    result={'representation':representation,'checkpoint':str(checkpoint),
        'episode_id':episode['id'],'current_index':observation,
        'future_observations_used':False,'inference_precision':'bf16','flow_steps':10}
    try:
        commands=policy.get_action()
        result.update(status='commands_decoded',command_shape=list(commands.shape))
        np.save(output/f'{representation}-commands.npy',commands)
    except ValueError as error:
        result.update(status='invalid_prediction',reason=str(error))
    torch.cuda.synchronize()
    result.update(first_call_observation_to_decoded_s=time.perf_counter()-start,
                  counts=policy.prediction_counts)
    if representation=='C' and hasattr(policy,'last_predicted_parameters'):
        np.savez(output/'C-prediction.npz',packed=policy.last_predicted_parameters,
            unscaled=policy.spline_parameter_scale.decode(policy.last_predicted_parameters))
    results.append(result)
    (output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    del policy
    gc.collect(); torch.cuda.empty_cache()
