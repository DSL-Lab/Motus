"""Warmup and optimizer/scheduler restoration with the actual scheduler class."""
import ast
from pathlib import Path
import unittest

import torch


class SchedulerContract(unittest.TestCase):
    def test_first_update_and_resume(self):
        path = Path(__file__).resolve().parents[1] / 'utils/scheduler.py'
        tree = ast.parse(path.read_text())
        namespace = {}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)
                                     and n.name == 'LambdaLinearScheduler'], type_ignores=[]), str(path), 'exec'), namespace)
        Scheduler = namespace['LambdaLinearScheduler']
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([parameter], lr=.001)
        scheduler = Scheduler(optimizer, 10, 100, f_start=.01)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .00001)
        parameter.sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer_state, scheduler_state = optimizer.state_dict(), scheduler.state_dict()
        restored_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=.001)
        restored_scheduler = Scheduler(restored_optimizer, 10, 100, f_start=.01)
        restored_optimizer.load_state_dict(optimizer_state)
        restored_scheduler.load_state_dict(scheduler_state)
        self.assertEqual(restored_optimizer.param_groups[0]['lr'], optimizer.param_groups[0]['lr'])
        scheduler.step()
        restored_scheduler.step()
        self.assertEqual(restored_scheduler.get_last_lr(), scheduler.get_last_lr())


if __name__ == '__main__':
    unittest.main()
