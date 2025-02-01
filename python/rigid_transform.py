#!/usr/bin/env python3

import numpy as np


# Custom exceptions to make it easier to distinguish in the unit tests
class SrcDstSizeMismatchError(Exception):
    pass


class InvalidPointDimError(Exception):
    pass


class NotEnoughPointsError(Exception):
    pass


class RankDeficiencyError(Exception):
    pass


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

    if src_pts.shape != dst_pts.shape:
        raise SrcDstSizeMismatchError(
            f"src and dst points aren't the same matrix size {src_pts.shape=} != {dst_pts.shape=}"
        )

    if not (dim == 2 or dim == 3):
        raise InvalidPointDimError(f"Points must be 2D or 3D, src_pts.shape[1] = {dim}")

    if src_pts.shape[0] < dim:
        raise NotEnoughPointsError(f"Not enough points, expect >= {dim} points")

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

    rank = np.linalg.matrix_rank(H)

    if dim == 2 and rank == 0:
        raise RankDeficiencyError(
            f"Insufficent matrix rank. For 2D points expect rank >= 1 but got {rank}. Maybe your points are all the same?"
        )
    elif dim == 3 and rank <= 1:
        raise RankDeficiencyError(
            f"Insufficent matrix rank. For 3D points expect rank >= 2 but got {rank}. Maybe your points are collinear?"
        )

    # find rotation
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    # https://en.wikipedia.org/wiki/Kabsch_algorithm
    det = np.linalg.det(R)
    if det < 0:
        print(f"det(R) = {det}, reflection detected!, correcting for it ...")
        S = np.eye(dim)
        S[-1, -1] = -1
        R = Vt.T @ S @ U.T

    if calc_scale:
        scale = np.sqrt(np.mean(dst_pts**2) / np.mean(src_pts**2))
    else:
        scale = 1.0

    t = -scale * R @ centroid_src.T + centroid_dst.T

    return R, t, scale
