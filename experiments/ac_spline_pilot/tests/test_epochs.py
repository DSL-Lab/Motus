"""Exact epoch stopping, partial accumulation, and final producer checkpoint."""
import logging
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import typing
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin
from omegaconf import OmegaConf

from experiments.ac_spline_pilot.tests.test_research import definitions


def trainer_class():
    ns=dict(torch=torch,Dict=typing.Dict,Any=typing.Any,Optional=typing.Optional,
            OmegaConf=OmegaConf,Motus=nn.Module,DataLoader=torch.utils.data.DataLoader,
            Path=Path,logger=logging.getLogger('epoch-test'),time=time)
    definitions('train/train.py',{'UniDiffuserTrainer'},ns)
    return ns['UniDiffuserTrainer']


class EpochTests(unittest.TestCase):
    def test_exact_passes_flush_one_partial_group_with_correct_gradient_mean(self):
        for windows,epochs in [(3,15),(4,2)]:
            with self.subTest(windows=windows,epochs=epochs):
                visited=[]
                class Toy(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.weight=nn.Parameter(torch.zeros(1))
                    def forward(self,first_frame,actions,**kwargs):
                        visited.append(int(first_frame.item()))
                        loss=(self.weight-actions).square().mean()
                        return dict(total_loss=loss,action_loss=loss,video_loss=loss*0)
                batches=[dict(first_frame=torch.tensor([[i]]),video_frames=torch.zeros(1,8,3,1,1),
                              action_sequence=torch.full((1,1),2.),vlm_inputs=None,language_embedding=None)
                         for i in range(windows)]
                accelerator=Accelerator(cpu=True,gradient_accumulation_plugin=GradientAccumulationPlugin(num_steps=4,sync_with_dataloader=False))
                model=Toy()
                optimizer=torch.optim.SGD(model.parameters(),lr=.1)
                loader=torch.utils.data.DataLoader(batches,batch_size=1,shuffle=True,collate_fn=lambda rows:rows[0])
                model,optimizer,loader=accelerator.prepare(model,optimizer,loader)
                cls=trainer_class()
                trainer=cls.__new__(cls)
                trainer.model,trainer.optimizer,trainer.accelerator=model,optimizer,accelerator
                trainer.device,trainer.dtype=torch.device('cpu'),torch.float32
                trainer.train_dataloader,trainer.val_dataloader=loader,None
                trainer.config=OmegaConf.create(dict(training=dict(grad_clip_norm=100.)))
                trainer.scheduler=Mock()
                trainer.global_step=0
                trainer.rank=0
                trainer.log_interval=trainer.save_interval=1000
                trainer.save_checkpoint=Mock()
                steps=(epochs*windows+3)//4
                trainer.train(max_steps=steps,max_epochs=epochs)
                self.assertEqual(trainer.global_step,steps)
                self.assertEqual(trainer.microbatches_seen,windows*epochs)
                self.assertEqual(trainer.epoch,epochs)
                self.assertEqual(trainer.scheduler.step.call_count,steps)
                for start in range(0,len(visited),windows):
                    self.assertEqual(sorted(visited[start:start+windows]),list(range(windows)))
                expected=2*(1-.8**steps)
                torch.testing.assert_close(accelerator.unwrap_model(model).weight.detach(),torch.tensor([expected]))
                trainer.save_checkpoint.assert_called_once()
                with self.assertRaisesRegex(ValueError,'fresh run'):
                    trainer.train(max_steps=steps,max_epochs=epochs,resume_from='old-checkpoint')
                with self.assertRaisesRegex(ValueError,'budget disagrees'):
                    trainer.train(max_steps=steps-1,max_epochs=epochs)

    def test_final_checkpoint_is_published_from_actual_finished_step(self):
        ns=dict(OmegaConf=OmegaConf,Path=Path,UniDiffuserTrainer=object,ACSplineTrainer=object,
                load_research_bundle=Mock())
        definitions('train/train.py',{'TrainingFactory'},ns)
        ns['__name__']='experiments.ac_spline_pilot._epoch_test'
        definitions('experiments/ac_spline_pilot/trainer.py',{'ACSplineFactory'},ns)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            checkpoint=root/'checkpoint_step_1052'
            checkpoint.mkdir()
            (checkpoint/'research_adapter.pt').touch()
            (checkpoint/'research_config.json').write_text('{}')
            (root/'checkpoint_step_1000').mkdir()
            config=OmegaConf.create(dict(training=dict(max_steps=1052,num_epochs=15,scheduler_type='linear'),
                resume=dict(checkpoint_path=None),system=dict(checkpoint_dir=tmp,val_interval=1052)))
            trainer=Mock(rank=0,global_step=1052)
            with patch('experiments.ac_spline_pilot.diagnostic.generate_diagnostic'):
                factory=ns['ACSplineFactory']()
                factory.run(trainer,SimpleNamespace(diagnostic_eval_only=None),config)
                self.assertEqual((root/'final_checkpoint.txt').read_text(),'checkpoint_step_1052\n')
                trainer.train.assert_called_once_with(max_steps=1052,resume_from=None,val_interval=1052,max_epochs=15)
                trainer.global_step=0
                factory.run(trainer,SimpleNamespace(diagnostic_eval_only='old-checkpoint'),config)
                self.assertEqual((root/'final_checkpoint.txt').read_text(),'checkpoint_step_1052\n')


if __name__=='__main__': unittest.main()
