import numpy as np
from absl.testing import absltest
from recon.datasets import data_augmentation


class DataAugmentationTest(absltest.TestCase):
    def test_rotations_3d(self):
        x = np.arange(8).reshape((2, 2, 2))
        rots = []
        for i in range(24):
            rotx = data_augmentation.apply_rotation(x, i)
            self.assertEqual(rotx.shape, x.shape)
            rots.append(str(rotx.flatten()))
        # There should be 24 distinct rotations.
        self.assertEqual(len(set(rots)), 24)
        self.assertEqual(data_augmentation.NUM_ROTATIONS, 24)

    def test_isometries_3d(self):
        x = np.arange(8).reshape((2, 2, 2))
        isos = []
        for i in range(48):
            isox = data_augmentation.apply_isometry(x, i)
            self.assertEqual(isox.shape, x.shape)
            isos.append(str(isox.flatten()))
        # There should be 48 distinct isometries.
        self.assertEqual(len(set(isos)), 48)
        self.assertEqual(data_augmentation.NUM_ISOMETRIES, 48)

    def test_rotations_4d(self):
        x = np.arange(24).reshape((2, 2, 2, 3))
        rots = []
        for i in range(24):
            rotx = data_augmentation.apply_rotation(x, i)
            self.assertEqual(rotx.shape, x.shape)
            rots.append(str(rotx.flatten()))
        # There should be 24 distinct rotations.
        self.assertEqual(len(set(rots)), 24)
        self.assertEqual(data_augmentation.NUM_ROTATIONS, 24)

    def test_isometries_4d(self):
        x = np.arange(24).reshape((2, 2, 2, 3))
        isos = []
        for i in range(48):
            isox = data_augmentation.apply_isometry(x, i)
            self.assertEqual(isox.shape, x.shape)
            isos.append(str(isox.flatten()))
        # There should be 48 distinct isometries.
        self.assertEqual(len(set(isos)), 48)
        self.assertEqual(data_augmentation.NUM_ISOMETRIES, 48)


if __name__ == "__main__":
    absltest.main()
