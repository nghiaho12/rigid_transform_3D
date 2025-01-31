#!/usr/bin/env python3

import numpy as np


def rigid_transform(src_pts, dst_pts, calc_scale=False):
    """Calculates the optimal rigid transform from src_pts to dst_pts.

    The returned transform minimizes the following least-squares problem
        r = dst_pts - (R @ src_pts + t)
        s = sum(r**2))

    If calc_scale is True, the similarity transform is solved, with the residual being
        r = dst_pts - (scale * R @ src_pts + t)
    where scale is a scalar.

    Parameters
    ----------
    src_pts: matrix of points stored as rows (e.g. Nx3)
    dst_pts: matrix of points stored as rows (e.g. Nx3)
    calc_scale: if True solve for scale

    Returns
    -------
    R: rotation matrix
    t: translation column vector
    scale: scalar, scale=1.0 if calc_scale=False
    """

    dim = src_pts.shape[1]

    assert dim == 2 or dim == 3, "dim must be 2 or 3"
    assert (
        src_pts.shape == dst_pts.shape
    ), f"src and dst points aren't the same shape {src_pts.shape=} != {dst_pts.shape=}"
    assert src_pts.shape[1] == dim, f"Expect Nx{dim} matrix of points"

    assert src_pts.shape[0] >= dim, f"Not enough points, expect >= {dim}"

    # find mean/centroid
    centroid_src = np.mean(src_pts, axis=0)
    centroid_dst = np.mean(dst_pts, axis=0)

    centroid_src = centroid_src.reshape(-1, dim)
    centroid_dst = centroid_dst.reshape(-1, dim)

    # subtract mean
    # NOTE: doing src_pts -= centroid_src will modifiy input!
    src_pts = src_pts - centroid_src
    dst_pts = dst_pts - centroid_dst

    # the cross-covariance matrix minus the mean calculation for each element
    # https://en.wikipedia.org/wiki/Cross-covariance_matrix
    H = src_pts.T @ dst_pts

    assert H.shape[0] == dim and H.shape[1] == dim, f"H matrix is not {dim}x{dim}"

    # sanity check
    rank = np.linalg.matrix_rank(H)
    assert rank == dim, f"Insufficent matrix rank, expect {dim} but got {rank}"

    # find rotation
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    if np.linalg.det(R) < 0:
        print("det(R) < 1, reflection detected!, correcting for it ...")
        S = np.eye(dim)
        S[-1] = -1
        R = Vt.T @ S @ U.T

    if calc_scale:
        scale = np.sqrt(np.mean(dst_pts**2) / np.mean(src_pts**2))
    else:
        scale = 1.0

    t = -scale * R @ centroid_src.T + centroid_dst.T

    return R, t, scale
