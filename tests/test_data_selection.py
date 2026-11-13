"""Episode-count requests must not silently use the old 50-demo cap."""
from pathlib import Path
import tempfile
import unittest

from motus_actions.prepare import select_episode_files


class EpisodeSelectionTests(unittest.TestCase):
    def test_five_hundred_distinct_episodes_are_selected_numerically(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for index in range(510): (root/f'episode_{index}.hdf5').touch()
            selected=select_episode_files(root,500)
            self.assertEqual(len(selected),500)
            self.assertEqual(len(set(selected)),500)
            self.assertEqual(selected[-1].stem,'episode_499')
            self.assertEqual(selected[10].stem,'episode_10')

    def test_fifty_is_not_accepted_as_five_hundred(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for index in range(50): (root/f'episode_{index}.hdf5').touch()
            with self.assertRaisesRegex(ValueError,'requested 500 demonstrations, found 50'):
                select_episode_files(root,500)
            self.assertEqual(len(select_episode_files(root,50)),50)


if __name__=='__main__': unittest.main()
