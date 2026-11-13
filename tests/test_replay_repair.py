"""Independent controller instrumentation checks; no simulator or large model."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from allocated_replay import execute_measured,compare_scene,assess_replay


class ReplayTests(unittest.TestCase):
    def test_acceptance_requires_five_pairs_and_a_predeclared_tracking_tolerance(self):
        import copy
        episode=dict(status='executed',offline_metrics=dict(max_normalized_reconstruction_error=.005,
            additional_transition_jump_max=.005,gripper_event_count_direction_match=True,
            gripper_transition_max_error_source_samples=.5),executions=[dict(variant=v,status='completed',
            success=True,tracking_max_abs_rad=.02) for v in ('original_full','original_matched','spline_matched')])
        rows=[copy.deepcopy(episode) for _ in range(5)]
        self.assertEqual(assess_replay(rows,.1),dict(original_replay='passed',spline_replay='passed',matched_controller='passed'))
        self.assertEqual(assess_replay(rows,None)['original_replay'],'blocked')
        self.assertEqual(assess_replay(rows[:1],.1)['original_replay'],'blocked')
        rows[0]['executions'][0]['success']=False
        gates=assess_replay(rows,.1)
        self.assertEqual(gates['original_replay'],'failed')
        self.assertEqual(gates['spline_replay'],'blocked')

    def test_physics_measurements_are_not_gripper_drive_targets(self):
        targets={'left':0.,'right':0.}
        physical={'left':0.,'right':0.}
        joint=SimpleNamespace(get_limits=lambda:np.array([[-2.,2.]]))
        robot=SimpleNamespace(left_arm_joints=[joint]*6,right_arm_joints=[joint]*6,
            get_left_arm_real_jointState=lambda:[0.]*6+[targets['left']],
            get_right_arm_real_jointState=lambda:[0.]*6+[targets['right']],
            set_arm_joints=lambda *args:None,set_gripper=lambda value,side:targets.update({side:value}))
        for side in ('left','right'):
            finger=object()
            setattr(robot,f'{side}_entity',SimpleNamespace(
                get_qpos=lambda side=side:np.array([physical[side]]),
                get_active_joints=lambda finger=finger:[finger]))
            setattr(robot,f'{side}_gripper',[(finger,1.,0.)])
            setattr(robot,f'{side}_gripper_scale',[0.,1.])
        steps=[]
        def step():
            steps.append(1)
            for side in ('left','right'):
                physical[side]+=np.clip(targets[side]-physical[side],-.2,.2)
        env=SimpleNamespace(robot=robot,scene=SimpleNamespace(get_timestep=lambda:.004,step=step),
            check_success=lambda:len(steps)>=3)
        actions=np.zeros((2,14)); actions[:,[6,13]]=1.
        plan=SimpleNamespace(actions=actions,control_dt=.012,execution_times=np.array([0.,.012]),requested_duration=.024)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trace.npz'
            result=execute_measured(env,plan,path)
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['physics_steps'],6)
            self.assertEqual(result['action_commands'],2)
            self.assertTrue(result['success'])
            with np.load(path) as data:
                self.assertEqual(len(data['measured_states']),7)
                self.assertAlmostEqual(data['measured_states'][1,6],.2)
                self.assertEqual(data['commands'][0,6],1.)
            self.assertAlmostEqual(result['measured_gripper_events'][0][0]['time_s'],.010)

    def test_scene_identity_requires_velocity_and_all_actors(self):
        compare_scene({'a':[0.,1.]},{'a':[0.,1.]})
        with self.assertRaisesRegex(ValueError,'physics state'):
            compare_scene({'a':[0.,1.]},{'a':[0.,2.]})
        with self.assertRaisesRegex(ValueError,'physics state'):
            compare_scene({'a':[0.,1.]},{'a':[0.,1.],'b':[0.]})


if __name__=='__main__': unittest.main()
