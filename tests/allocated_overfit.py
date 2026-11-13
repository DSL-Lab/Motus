"""Fixed-example/fixed-noise overfit using the existing trainer (not a pilot).

Acceptance declared before runs: 50 updates reduce fixed loss by >=20%; all
trainable gradients finite; native full checkpoint reload exactly restores
trainable tensors, optimizer, scheduler and torch RNG. Batch1 is diagnostic only.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'train'))
from train import UniDiffuserTrainer, create_model_and_optimizer, create_accelerator, prepare_pilot_training
from data.dataset import create_dataset, collate_fn


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(cpu_state(v) for v in value)
    return copy.deepcopy(value)


def assert_state_equal(actual,expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
    elif isinstance(expected,np.ndarray):
        np.testing.assert_array_equal(actual,expected)
    elif isinstance(expected,dict):
        if actual.keys()!=expected.keys(): raise ValueError('resume state keys differ')
        for key in expected: assert_state_equal(actual[key],expected[key])
    elif isinstance(expected,(tuple,list)):
        if len(actual)!=len(expected): raise ValueError('resume state length differs')
        for a,b in zip(actual,expected): assert_state_equal(a,b)
    elif actual!=expected:
        raise ValueError(f'resume state mismatch: {actual} != {expected}')


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def varied_main(representation,output,until,resume,lora_rank=16):
    """Small varied-noise subset through the existing trainer; separate CLI processes resume."""
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('requires a Slurm GPU allocation')
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    config=OmegaConf.load(ROOT/'configs/robotwin_ac.yaml')
    config.training.lora_rank=lora_rank  # Default repair phase 16; explicit follow-up can use 256.
    config.common.action_representation='dense' if representation=='A' else 'spline'
    config.training.max_steps=256  # fixed schedule, also during the short resume stages
    config.training.warmup_steps=13
    config.training.gradient_accumulation_steps=1  # diagnostic only, pilot remains 16
    config.system.checkpoint_dir=str(output)
    if resume is None:
        output.mkdir(exist_ok=False)
        dataset=create_dataset(config)
        samples=[dataset[0] for _ in range(4)]
        validation=[create_dataset(config,val=True)[0]]
        torch.save(dict(train=samples,validation=validation),output/'examples.pt')
    else:
        selected=torch.load(output/'examples.pt',weights_only=False)
        samples,validation=selected['train'],selected['validation']
    OmegaConf.save(config,output/'resolved_config.yaml')
    loader=DataLoader(samples,batch_size=1,shuffle=True,collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(0))
    model,optimizer,scheduler=create_model_and_optimizer(config)
    history=[]
    original=model.training_step
    def record_step(**kwargs):
        result=original(**kwargs)
        history.append({k:float(v.detach()) if torch.is_tensor(v) else v for k,v in result.items()})
        return result
    model.training_step=record_step
    accelerator=create_accelerator(1,mixed_precision='bf16')
    model,optimizer,loader,scheduler=prepare_pilot_training(accelerator,model,optimizer,loader,scheduler)
    trainer=UniDiffuserTrainer(model,loader,optimizer=optimizer,scheduler=scheduler,device=accelerator.device,
        checkpoint_dir=str(output),log_interval=16,save_interval=128,val_interval=1000,
        report_to=[],accelerator=accelerator,config=config)
    restored=None
    if resume:
        trainer.load_checkpoint(str(resume),reset_scheduler=False)
        expected=torch.load(resume/'resume_audit.pt',weights_only=False)
        assert_state_equal(rng_state(),expected['rng'])
        assert_state_equal({n:p for n,p in model.named_parameters() if p.requires_grad},expected['trainable'])
        assert_state_equal(optimizer.state_dict(),expected['optimizer'])
        assert_state_equal(scheduler.state_dict(),expected['scheduler'])
        if trainer.global_step!=expected['global_step']:
            raise ValueError('global step not restored')
        restored=dict(status='passed',checkpoint=str(resume),fresh_process=True,pid=os.getpid(),
            previous_pid=expected['pid'],global_step=trainer.global_step,
            trainable_parameters='bit_exact',optimizer='bit_exact',scheduler='bit_exact',
            python_numpy_torch_cpu_cuda_rng='bit_exact',accumulation='safe_update_boundary')
        if restored['pid']==expected['pid']: raise ValueError('resume did not restart the process')
        (output/f'resume-from-{trainer.global_step}.json').write_text(json.dumps(restored,indent=2)+'\n')
    torch.cuda.synchronize(); start=time.perf_counter()
    trainer.train(max_steps=until,val_interval=1000)
    torch.cuda.synchronize()
    checkpoint=output/f'checkpoint_step_{until}'
    torch.save(dict(trainable=cpu_state({n:p for n,p in model.named_parameters() if p.requires_grad}),
        optimizer=cpu_state(optimizer.state_dict()),scheduler=cpu_state(scheduler.state_dict()),
        global_step=trainer.global_step,rng=rng_state(),pid=os.getpid()),checkpoint/'resume_audit.pt')
    result=dict(representation=representation,lora_rank=lora_rank,updates=trainer.global_step,mode='varied_noise_and_flow_times',
        examples=[dict(episode=s['episode_id'],observation=s['observation_index']) for s in samples],
        elapsed_s=time.perf_counter()-start,history=history,resume=restored)
    if until>=256:
        result['generation']=evaluate_subset(model,config,samples,validation,output,representation)
    (output/f'varied-step-{until}.json').write_text(json.dumps(result,indent=2)+'\n')


def evaluate_subset(model,config,samples,validation,output,representation):
    from inference.robotwin.Motus.deploy_policy import MotusPolicy
    from allocated_decode import decoder_policy,knot_record,trajectory_metrics
    from collections import deque
    from utils.robotwin_prompt import robotwin_prompt
    manifest=json.loads(Path(config.dataset.episode_manifest).read_text())
    policy=decoder_policy(manifest)
    policy.config_dict=OmegaConf.to_container(config,resolve=True)
    policy.model=model.eval()
    policy.device='cuda'
    policy.experiment_lora=True
    policy.spline=representation=='C'
    policy.save_images=False
    policy.step_count=0
    policy.vlm_processor=create_dataset(config).vlm_processor
    all_records=[]
    for split,selected in [('training',samples),('held_out',validation)]:
        for index,sample in enumerate(selected):
            entry=next(e for e in manifest['episodes'] if e['id']==sample['episode_id'])
            with np.load(entry['cache_path']) as cached:
                truth=cached['actions'][sample['observation_index']+np.arange(1,17)*3]
            for seed in (13,29,47):
                policy.prediction_counts=dict(knot_repairs=0,repairs=0,invalid_predictions=0,short_coverage=0,fallbacks=0)
                policy.obs_cache,policy.action_cache=deque(maxlen=1),deque()
                def cached_t5(text,device):
                    if text!=[sample['text_instruction']]: raise ValueError('evaluation conditioning differs')
                    return [sample['language_embedding'].to(device)]
                policy.t5_encoder=cached_t5
                policy.set_instruction(entry['instruction'])
                observation=dict(image=(sample['first_frame'].permute(1,2,0).numpy()*255).round().astype(np.uint8),
                    joint_action={'vector':sample['initial_state'].numpy()})
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize(); start=time.perf_counter()
                record=dict(split=split,example=index,episode=entry['id'],observation=sample['observation_index'],noise_seed=seed)
                try:
                    policy.update_obs(observation)
                    commands=policy.get_action()
                    record.update(status='decoded',decoder_failures=0,execution_coverage=1.,
                        **trajectory_metrics(commands,truth,policy.spline_action_scale,np.arange(1,17)*3))
                except ValueError as error:
                    record.update(status='decoder_failure',decoder_failures=1,reason=str(error),
                        trajectory_normalized_rmse=None,endpoint_normalized_max=None,
                        gripper_transition_max_error_samples=None,execution_coverage=0.)
                torch.cuda.synchronize()
                record.update(offline_observation_to_decode_s=time.perf_counter()-start,counts=policy.prediction_counts.copy())
                if policy.spline:
                    record.update(raw=knot_record(policy.last_raw_parameters),postprocessed=knot_record(policy.last_postprocessed_parameters))
                    arrays=dict(packed=policy.last_predicted_parameters)
                    if policy.last_raw_parameters is not None: arrays['raw']=policy.last_raw_parameters
                    if policy.last_postprocessed_parameters is not None: arrays['postprocessed']=policy.last_postprocessed_parameters
                    np.savez(output/f'{split}-{index}-seed-{seed}.npz',**arrays)
                all_records.append(record)
                (output/'generation.json').write_text(json.dumps(all_records,indent=2)+'\n')
    training=[r for r in all_records if r['split']=='training']
    passed=all(r['status']=='decoded' and r['trajectory_normalized_rmse']<=.05 and
        r['endpoint_normalized_max']<=.1 and r['gripper_event_count_direction_match'] and
        r['gripper_transition_max_error_samples']<=3 for r in training)
    summary=dict(training_generation_passed=passed,training_calls=len(training),held_out_calls=len(all_records)-len(training),
        raw_knot_valid=sum(r['raw']['valid'] for r in training) if representation=='C' else None,
        postprocessed_knot_valid=sum(r['postprocessed']['valid'] for r in training) if representation=='C' else None,
        decoder_failures=sum(r['decoder_failures'] for r in training),
        repairs=sum(r['counts']['repairs'] for r in training),fallbacks=0)
    (output/'generation-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main(representation):
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available():
        raise RuntimeError('requires a Slurm GPU allocation')
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    config = OmegaConf.load(ROOT/'configs/robotwin_ac.yaml')
    config.training.lora_rank = 16
    config.common.action_representation = 'dense' if representation == 'A' else 'spline'
    config.training.max_steps = 50
    config.training.warmup_steps = 3
    config.training.gradient_accumulation_steps = 1
    output = ROOT/'outputs'/f'tiny-{representation}-{os.environ["SLURM_JOB_ID"]}'
    output.mkdir(exist_ok=False)
    config.system.checkpoint_dir = str(output)
    OmegaConf.save(config, output/'resolved_config.yaml')
    sample = create_dataset(config)[0]
    loader = DataLoader([sample], batch_size=1, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(0))
    model, optimizer, scheduler = create_model_and_optimizer(config)
    # Freeze the flow draw in this numerical overfit check only.
    original_step = model.training_step
    history = []
    def fixed_step(**kwargs):
        torch.manual_seed(71)
        torch.cuda.manual_seed_all(71)
        result = original_step(**kwargs)
        history.append(float(result['total_loss'].detach()))
        return result
    model.training_step = fixed_step
    accelerator = create_accelerator(1,mixed_precision='bf16')
    model,optimizer,loader,scheduler=prepare_pilot_training(accelerator,model,optimizer,loader,scheduler)
    trainer = UniDiffuserTrainer(model, loader, optimizer=optimizer, scheduler=scheduler,
        device=accelerator.device, checkpoint_dir=str(output), log_interval=1,
        save_interval=25, val_interval=1000, report_to=[], accelerator=accelerator, config=config)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize(); start = time.perf_counter()
    trainer.train(max_steps=50, val_interval=1000)
    torch.cuda.synchronize(); elapsed = time.perf_counter()-start
    parameters = {n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    rng_expected = torch.randn(8,device='cuda')
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad: p.add_(.01)
    accelerator.load_state(str(output/'checkpoint_step_50'))
    for name,p in model.named_parameters():
        if p.requires_grad: torch.testing.assert_close(p, parameters[name], rtol=0,atol=0)
    torch.testing.assert_close(torch.randn(8,device='cuda'),rng_expected,rtol=0,atol=0)
    if scheduler.state_dict() != scheduler_state or not optimizer.state_dict()['state']:
        raise ValueError('optimizer/scheduler resume failure')
    result = dict(representation=representation, updates=50, microbatch=1, fixed_flow_seed=71,
        initial_loss=history[0], final_loss=history[-1], losses=history,
        acceptance_ratio=.8, overfit_passed=history[-1] <= .8*history[0],
        full_checkpoint_reload='passed', elapsed_s=elapsed,
        peak_memory_bytes=torch.cuda.max_memory_allocated())
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    # Current-observation-only inference; never supply demonstration future frames.
    model.eval()
    batch = collate_fn([sample])
    inference = dict(first_frame=batch['first_frame'].cuda().bfloat16(),
        state=batch['initial_state'].cuda().bfloat16(), num_inference_steps=10,
        language_embeddings=batch['language_embedding'].cuda().bfloat16(),
        vlm_inputs={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch['vlm_inputs'].items()})
    with torch.no_grad():
        torch.cuda.manual_seed_all(13)
        _, prediction = model.inference_step(**inference)
    if not torch.isfinite(prediction).all(): raise ValueError('nonfinite inference')
    result['inference_shape']=list(prediction.shape)
    if representation == 'C':
        from motus_actions.spline import AffineScale,decode
        manifest = json.loads(Path(config.dataset.episode_manifest).read_text())
        try:
            local=AffineScale(manifest['parameter_low'],manifest['parameter_high']).decode(prediction[0].cpu().numpy())
            commands=AffineScale(manifest['action_low'],manifest['action_high']).decode(decode(local,np.arange(1,17)*3))
            result['controller_decode']='passed'
            result['controller_command_shape']=list(commands.shape)
        except ValueError as error:
            result['controller_decode']='invalid_prediction'
            result['invalid_predictions']=1
            result['decode_reason']=str(error)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='losses'}),flush=True)
    if not result['overfit_passed']: raise SystemExit(2)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('representation',choices=['A','C'])
    parser.add_argument('--varied',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--until',type=int,default=256)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--lora-rank',type=int,default=16)
    args=parser.parse_args()
    if args.varied:
        if args.output is None: parser.error('--output required for varied diagnostic')
        varied_main(args.representation,args.output,args.until,args.resume,args.lora_rank)
    else:
        main(args.representation)
