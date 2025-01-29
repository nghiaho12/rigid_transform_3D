#!/usr/bin/env python3

import numpy as np
import unittest
from scipy.spatial.transform import Rotation
from rigid_transform_3D import rigid_transform_3D


def random_points(n=100):
    return np.random.rand(n, 3)


def random_rotation():
    R = Rotation.from_rotvec(np.random.rand(3)).as_matrix()

    # remove reflection
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    if np.linalg.det(R) < 0:
        U, _, Vt = np.linalg.svd(R)
        S = np.diag([1, 1, -1])
        R = U @ S @ Vt

    return R


def random_translation():
    return np.random.rand(3, 1)


class Test(unittest.TestCase):
    def test_bad_input(self):
        src_pts = random_points(100)
        dst_pts = random_points(50)

        with self.assertRaises(AssertionError):
            rigid_transform_3D(src_pts, dst_pts)

    def test_Nx3(self):
        pts = random_points(100)
        R, t = rigid_transform_3D(pts, pts)

        self.assertAlmostEqual(np.sum((R - np.eye(3)) ** 2), 0.0)
        self.assertAlmostEqual(np.sum(t**2), 0.0)

    def test_3xN(self):
        pts = random_points(100).T
        R, t = rigid_transform_3D(pts, pts)

        self.assertAlmostEqual(np.sum((R - np.eye(3)) ** 2), 0.0)
        self.assertAlmostEqual(np.sum(t**2), 0.0)

    def test(self):
        R = random_rotation()
        t = random_translation()
        src = random_points()
        dst = R @ src.T + t

        # Recover R and t
        ret_R, ret_t = rigid_transform_3D(src, dst.T)
        dst2 = ret_R @ src.T + ret_t

        rmse = np.sqrt(np.mean(((dst - dst2) ** 2).flatten()))

        print("Ground truth rotation")
        print(R)

        print("Recovered rotation")
        print(ret_R)
        print("")

        print("Ground truth translation: ", t.flatten())
        print("Recovered translation: ", ret_t.flatten())
        print("")
        print("RMSE:", rmse)

        if rmse < 1e-5:
            print("Everything looks good!")
        else:
            print("Hmm something doesn't look right ...")

        self.assertAlmostEqual(rmse, 0.0)


if __name__ == "__main__":
    unittest.main()
