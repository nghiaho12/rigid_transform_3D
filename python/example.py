#!/usr/bin/env python3

import numpy as np
from scipy.spatial.transform import Rotation
from rigid_transform import rigid_transform


def random_points(n, dim):
    return np.random.rand(n, dim)


def random_rotation(dim):
    if dim == 2:
        R = Rotation.from_euler("z", np.random.randn()).as_matrix()[:2, :2]
    elif dim == 3:
        R = Rotation.from_rotvec(np.random.rand(3)).as_matrix()

        # remove reflection
        # https://en.wikipedia.org/wiki/Kabsch_algorithm
        if np.linalg.det(R) < 0:
            U, _, Vt = np.linalg.svd(R)
            S = np.diag([1, 1, -1])
            R = U @ S @ Vt
    else:
        raise ValueError("dim must be 2 or 3")

    return R


def random_translation(dim):
    return np.random.rand(dim, 1)


def random_scale():
    while True:
        s = np.random.rand(1) * 10
        if abs(s) > 0.1:
            return s.item()


if __name__ == "__main__":
    for dim in [2, 3]:
        print("=" * 60)
        print(f"{dim}D points")
        print("")

        R = random_rotation(dim)
        t = random_translation(dim)
        s = random_scale()
        src = random_points(100, dim)
        dst = s * src @ R.T + t.T

        # Recover R and t
        ret_R, ret_t, ret_s = rigid_transform(src, dst, calc_scale=True)
        dst2 = ret_s * src @ R.T + ret_t.T

        rmse = np.sqrt(np.mean(((dst - dst2) ** 2).flatten()))

        print("Ground truth rotation")
        print(R)
        print("")
        print("Recovered rotation")
        print(ret_R)
        print("")

        print("Ground truth translation: ", t.flatten())
        print("Recovered translation: ", ret_t.flatten())
        print("")
        print("Ground truth scale: ", s)
        print("Recovered scale: ", ret_s)
        print("")
        print("RMSE:", rmse)

        if rmse < 1e-6:
            print("Everything looks good!")
        else:
            print("Hmm something doesn't look right ...")
