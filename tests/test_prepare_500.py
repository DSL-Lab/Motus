"""ZIP inventory validation without extracting or converting real demonstrations."""
import importlib.util
import io
from pathlib import Path
import unittest
import zipfile

path = Path(__file__).resolve().parents[1]/'motus_actions/prepare_500.py'
spec = importlib.util.spec_from_file_location('prepare_500_inventory', path)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


class PreparationInventoryTests(unittest.TestCase):
    def archive(self, count=500, extra=None):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            prefix = preparation.ARCHIVE_ROOT
            for index in range(count):
                archive.writestr(f'{prefix}/data/episode{index}.hdf5', b'')
            archive.writestr(f'{prefix}/seed.txt', b'')
            archive.writestr(f'{prefix}/scene_info.json', b'{}')
            if extra:
                archive.writestr(extra, b'')
        stream.seek(0)
        return zipfile.ZipFile(stream)

    def test_requires_all_500_distinct_episode_indices(self):
        with self.archive() as archive:
            self.assertEqual(len(preparation.archive_members(archive)), 502)
        with self.archive(count=50) as archive:
            with self.assertRaisesRegex(ValueError, '0 through 499'):
                preparation.archive_members(archive)

    def test_rejects_unsafe_members_before_extraction(self):
        with self.archive(extra=f'{preparation.ARCHIVE_ROOT}/../outside') as archive:
            with self.assertRaisesRegex(ValueError, 'unsafe'):
                preparation.archive_members(archive)

    def test_rejects_additional_hdf5_layouts(self):
        with self.archive(extra=f'{preparation.ARCHIVE_ROOT}/other/episode0.hdf5') as archive:
            with self.assertRaisesRegex(ValueError, 'unexpected raw HDF5'):
                preparation.archive_members(archive)


if __name__ == '__main__':
    unittest.main()
