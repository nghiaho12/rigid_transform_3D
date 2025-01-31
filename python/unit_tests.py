#!/usr/bin/env python3

import numpy as np
import unittest
from rigid_transform import rigid_transform
from example import random_points, random_rotation, random_translation, random_scale


class Test(unittest.TestCase):
    def test_2d_points(self):
        dim = 2
        R = random_rotation(dim)
        t = random_translation(dim)
        scale = random_scale()

        src = random_points(100, dim)
        dst = scale * (src @ R.T) + t.T

        ret_R, ret_t, ret_scale = rigid_transform(src, dst, calc_scale=True)

        self.assertTrue(np.allclose(R, ret_R))
        self.assertTrue(np.allclose(t, ret_t))
        self.assertAlmostEqual(scale, ret_scale)

    def test_3d_points(self):
        dim = 3
        R = random_rotation(dim)
        t = random_translation(dim)
        scale = random_scale()

        src = random_points(100, dim)
        dst = scale * (src @ R.T) + t.T

        ret_R, ret_t, ret_scale = rigid_transform(src, dst, calc_scale=True)

        self.assertTrue(np.allclose(R, ret_R))
        self.assertTrue(np.allclose(t, ret_t))
        self.assertAlmostEqual(scale, ret_scale)

    def test_no_scale(self):
        dim = 3

        R = random_rotation(dim)
        t = random_translation(dim)

        src = random_points(100, dim)
        dst = (src @ R.T) + t.T

        ret_R, ret_t, ret_scale = rigid_transform(src, dst, calc_scale=False)

        self.assertTrue(np.allclose(R, ret_R))
        self.assertTrue(np.allclose(t, ret_t))
        self.assertAlmostEqual(ret_scale, 1.0)

    def test_invalid_dim(self):
        dim = 4
        src = random_points(100, dim)

        with self.assertRaises(AssertionError):
            rigid_transform(src, src, calc_scale=True)

    def test_mismatch_size(self):
        src = random_points(100, 2)
        dst = random_points(100, 3)

        with self.assertRaises(AssertionError):
            rigid_transform(src, dst, calc_scale=True)

    def test_wrong_order(self):
        src = random_points(3, 100)

        with self.assertRaises(AssertionError):
            rigid_transform(src, src, calc_scale=True)

    def test_not_enough_points(self):
        src = random_points(2, 3)

        with self.assertRaises(AssertionError):
            rigid_transform(src, src, calc_scale=True)

    def test_low_rank(self):
        src = np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])

        with self.assertRaises(AssertionError):
            rigid_transform(src, src, calc_scale=True)

    def test_reflection(self):
        src = np.array(
            [
                [0.04997603, 0.92769423],
                [0.64045334, 0.5110098],
                [0.80452329, 0.19618526],
            ]
        )

        dst = np.array(
            [
                [0.44224556, 0.61291874],
                [0.4559217, 0.17549108],
                [0.57923716, 0.23585888],
            ]
        )

        ret_R, ret_t, ret_scale = rigid_transform(src, dst, calc_scale=False)
        self.assertTrue(np.linalg.det(ret_R) == 1)


if __name__ == "__main__":
    unittest.main()
