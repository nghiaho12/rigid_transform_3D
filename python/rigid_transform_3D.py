#!/usr/bin/env python3

import numpy as np


def rigid_transform_3D(src_pts, dst_pts, calc_scale=False):
    """Calculates the optimal rigid transform from src_pts to dst_pts.

    The returned transform minimizes the following least-squares problem
        r = dst_pts - (R @ src_pts + t)
        s = sum(r**2))

    If calc_scale is True, the following residual is minimized instead
        r = dst_pts - (scale * R @ src_pts + t)

    where scale is a scalar.

    Parameters
    ------
    src_pts: 3xN or Nx3 matrix
    dst_pts: 3xN or Nx3 matrix
    calc_scale: bool

    NOTE: If src_pts and dst_pts are 3x3 matrices then points are assumed to be row major.

    Returns
    -------
    R: 3x3 rotation matrix
    t: 3x1 column vector
    scale: scalar, 1 if calc_scale=False
    """

    assert (
        src_pts.shape == dst_pts.shape
    ), f"src and dst points aren't the same shape {src_pts.shape=} != {dst_pts.shape=}"
    assert (
        src_pts.shape[0] == 3 or dst_pts.shape[1] == 3
    ), "Expect 3xN or Nx3 matrix of points"

    assert min(src_pts.shape) >= 3, "Not enough points, expect >= 3"

    # transpose to row major
    if src_pts.shape[0] == 3:
        src_pts = src_pts.T
        dst_pts = dst_pts.T

    # find mean/centroid
    centroid_src = np.mean(src_pts, axis=0)
    centroid_dst = np.mean(dst_pts, axis=0)

    centroid_src = centroid_src.reshape(-1, 3)
    centroid_dst = centroid_dst.reshape(-1, 3)

    # subtract mean
    # NOTE: doing src_pts -= centroid_src will modifiy input!
    src_pts = src_pts - centroid_src
    dst_pts = dst_pts - centroid_dst

    if calc_scale:
        scale = np.sqrt(np.sum(dst_pts**2) / np.sum(src_pts**2))
    else:
        scale = 1.0

    # almost the cross-covariance matrix, except the outer mean is not calculated
    # https://en.wikipedia.org/wiki/Cross-covariance_matrix
    H = src_pts.T @ dst_pts

    assert H.shape[0] == 3 and H.shape[1] == 3, "H is not a 3x3 matrix"

    # sanity check
    rank = np.linalg.matrix_rank(H)
    if rank < 3:
        print(f"WARNING: rank of H = {rank}, expecting rank of 3 for a unique solution")

    # find rotation
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    if np.linalg.det(R) < 0:
        print("det(R) < 1, reflection detected!, correcting for it ...")
        S = np.diag([1, 1, -1])
        R = Vt.T @ S @ U.T

    t = -scale * R @ centroid_src.T + centroid_dst.T

    return R, t, scale
