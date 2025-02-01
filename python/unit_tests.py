#!/usr/bin/env python3

import numpy as np
import unittest
from rigid_transform import (
    rigid_transform,
    SrcDstSizeMismatchError,
    InvalidPointDimError,
    NotEnoughPointsError,
    RankDeficiencyError,
)
from example import random_points, random_rotation, random_translation, random_scale


class Test(unittest.TestCase):
    def test_2d_not_enough_points(self):
        src = np.array([[1, 2]])

        with self.assertRaises(NotEnoughPointsError):
            rigid_transform(src, src)

    def test_2d_min_points(self):
        dim = 2

        R = random_rotation(dim)
        t = random_translation(dim)
        scale = random_scale()

        src = np.array([[1, 0], [-1, 0]])
        dst = scale * (src @ R.T) + t.T

        ret_R, ret_t, ret_scale = rigid_transform(src, dst, calc_scale=True)

        self.assertTrue(np.allclose(R, ret_R))
        self.assertTrue(np.allclose(t, ret_t))
        self.assertAlmostEqual(scale, ret_scale)

    def test_2d_points(self):
        dim = 2
        num_pts = 100

        R = random_rotation(dim)
        t = random_translation(dim)
        scale = random_scale()

        src = random_points(num_pts, dim)
        dst = scale * (src @ R.T) + t.T

        ret_R, ret_t, ret_scale = rigid_transform(src, dst, calc_scale=True)

        self.assertTrue(np.allclose(R, ret_R))
        self.assertTrue(np.allclose(t, ret_t))
        self.assertAlmostEqual(scale, ret_scale)

    def test_2d_rank_deficiency(self):
        src = np.array([[1, 1], [1, 1]])

        with self.assertRaises(RankDeficiencyError):
            rigid_transform(src, src)

    def test_2d_reflection(self):
        R = np.array([[-0.63360525, 0.7736565], [-0.7736565, -0.63360525]])
        t = np.array([[-1], [1]])
        scale = 1.23

        src = np.array([[1, 0], [-1, 0]])
        dst = scale * (src @ R.T) + t.T

        ret_R, ret_t, ret_scale = rigid_transform(src, dst, calc_scale=True)

        self.assertTrue(np.allclose(R, ret_R))
        self.assertTrue(np.allclose(t, ret_t))
        self.assertAlmostEqual(scale, ret_scale)

    def test_3d_not_enough_points(self):
        src = np.array([[1, 2, 0], [3, 3, 0]])

        with self.assertRaises(NotEnoughPointsError):
            rigid_transform(src, src)

    def test_3d_min_points(self):
        dim = 3

        R = random_rotation(dim)
        t = random_translation(dim)
        scale = random_scale()

        src = np.array([[1, 0, 0], [-1, 0, 0], [1, 1, 0]])
        dst = scale * (src @ R.T) + t.T

        ret_R, ret_t, ret_scale = rigid_transform(src, dst, calc_scale=True)

        self.assertTrue(np.allclose(R, ret_R))
        self.assertTrue(np.allclose(t, ret_t))
        self.assertAlmostEqual(scale, ret_scale)

    def test_3d_points(self):
        dim = 3
        num_pts = 100

        R = random_rotation(dim)
        t = random_translation(dim)
        scale = random_scale()

        src = random_points(num_pts, dim)
        dst = scale * (src @ R.T) + t.T

        self.assertTrue(np.allclose(R, ret_R))
        self.assertTrue(np.allclose(t, ret_t))
        self.assertAlmostEqual(scale, ret_scale)

    def test_3d_rank_deficiency(self):
        src = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

        with self.assertRaises(RankDeficiencyError):
            rigid_transform(src, src)

    def test_3d_reflection(self):
        R = np.array(
            [
                [0.66962123, 0.22398235, 0.7081238],
                [0.25805249, 0.8238756, -0.50461659],
                [-0.69643113, 0.52063509, 0.49388539],
            ]
        )
        t = np.array([[-1], [2], [-3]])
        scale = 2.34

        src = np.array([[1, 0, 0], [-1, 0, 0], [1, 1, 0]])
        dst = scale * (src @ R.T) + t.T

        ret_R, ret_t, ret_scale = rigid_transform(src, dst, calc_scale=True)

        self.assertTrue(np.allclose(R, ret_R))
        self.assertTrue(np.allclose(t, ret_t))
        self.assertAlmostEqual(scale, ret_scale)

    def test_calc_scale_false(self):
        dim = 3
        num_pts = 100

        R = random_rotation(dim)
        t = random_translation(dim)
        scale = 10.0

        src = random_points(num_pts, dim)
        dst = scale * (src @ R.T) + t.T

        ret_R, ret_t, ret_scale = rigid_transform(src, dst, calc_scale=False)

        self.assertTrue(np.allclose(R, ret_R))
        self.assertAlmostEqual(ret_scale, 1.0)

    def test_invalid_point_dim(self):
        dim = 4
        num_pts = 100

        src = random_points(num_pts, dim)

        with self.assertRaises(InvalidPointDimError):
            rigid_transform(src, src, calc_scale=True)

    def test_mismatch_size(self):
        num_pts = 100

        src = random_points(num_pts, 2)
        dst = random_points(num_pts, 3)

        with self.assertRaises(SrcDstSizeMismatchError):
            rigid_transform(src, dst, calc_scale=True)


if __name__ == "__main__":
    unittest.main()
