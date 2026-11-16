"""Actual HDF5 interchange contract: preserve action/state alignment and order."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from robotwin_spline.sidecar import load_actions


class RecordedActions(unittest.TestCase):
    def test_v1_actions_preserve_alignment_and_reject_wrong_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'episode.hdf5'
            values = np.arange(70, dtype=np.float32).reshape(5, 14)
            with h5py.File(path, 'w') as f:
                f.create_dataset('data_format_version', data='v1.0')
                f.create_dataset('state/joint_states', data=values - 10)
                f.create_dataset('action/joint_states', data=values)
                for name, region in [('left_arm_joint_states', slice(0,6)), ('left_ee_joint_states', slice(6,7)),
                                     ('right_arm_joint_states', slice(7,13)), ('right_ee_joint_states', slice(13,14))]:
                    f.create_dataset('action/' + name, data=values[:,region])
            np.testing.assert_array_equal(load_actions(path), values)
            with h5py.File(path, 'r+') as f:
                f['action/joint_states'][0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'ordering'):
                load_actions(path)


if __name__ == '__main__':
    unittest.main()
