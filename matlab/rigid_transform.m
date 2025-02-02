function [R, t, scale] = rigid_transform(src_pts, dst_pts, calc_scale=false)
    % Calculates the optimal rigid transform from src_pts to dst_pts.
    %
    % The returned transform minimizes the following least-squares problem
    %     r = dst_pts - (R @ src_pts + t)
    %     s = sum(r**2))
    %
    % If calc_scale is True, the similarity transform is solved, with the residual being
    %     r = dst_pts - (scale * R @ src_pts + t)
    % where scale is a scalar.
    %
    % Parameters
    % ----------
    % src_pts: matrix of points stored as rows (e.g. Nx3)
    % dst_pts: matrix of points stored as rows (e.g. Nx3)
    % calc_scale: if True solve for scale
    %
    % Returns
    % -------
    % R: rotation matrix
    % t: translation column vector
    % scale: scalar, scale=1.0 if calc_scale=False
    narginchk(2, 3);

    dim = size(src_pts, 2);

    assert(all(size(src_pts) == size(dst_pts)), sprintf("src and dst points aren't the same matrix size %dx%d != %dx%d", size(src_pts, 1), size(src_pts, 2), size(dst_pts, 1), size(dst_pts, 2)))

    assert(dim == 2 || dim == 3, sprintf("Points must be 2D or 3D, size(src_pts, 2) = %d", dim))

    assert(size(src_pts, 1) >= dim, sprintf("Not enough points, expect >= %d points", dim))

    % find mean/centroid
    centroid_src = mean(src_pts, 1);
    centroid_dst = mean(dst_pts, 1);

    % subtract mean
    src_pts = src_pts - centroid_src;
    dst_pts = dst_pts - centroid_dst;

    % this is similar to the cross-covariance matrix, except the outer mean is not calculated
    % https://en.wikipedia.org/wiki/Cross-covariance_matrix
    H = src_pts' * dst_pts;

    r = rank(H);

    if dim == 2 && r == 0
        error(sprintf("Insufficent matrix rank. For 2D points expect rank >= 1 but got %d. Maybe your points are all the same?", r))
    else if dim == 3 && r <= 1
        error(sprintf("Insufficent matrix rank. For 3D points expect rank >= 2 but got %d. Maybe your points are collinear?", r))
    end

    % find rotation
    [U, S, V] = svd(H);
    R = V * U';

    % special reflection case
    % https://en.wikipedia.org/wiki/Kabsch_algorithm
    if det(R) < 0
        printf("det(R) < 0, reflection detected!, correcting for it ...\n")
        S = eye(dim);
        S(dim, dim)  = -1;
        R = V * S * U';
    end

    if calc_scale
        scale = sqrt(mean(dst_pts.^2, "all") / mean(src_pts.^2, "all"));
    else
        scale = 1.0;
    end

    t = -scale * R * centroid_src' + centroid_dst';
end
