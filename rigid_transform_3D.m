function [R, t] = rigid_transform_3D(src_pts, dst_pts)
    % Calculates the optimal rigid transform from src_pts to dst_pts.
    %
    % Parameters
    % ------
    % src_pts: 3xN or Nx3 matrix
    % dst_pts: 3xN or Nx3 matrix
    %
    % NOTE: If src_pts and dst_pts are 3x3 matrices then points are assumed to be row major.
    %
    % Returns
    % -------
    % R = 3x3 rotation matrix
    % t = 3x1 column vector

    narginchk(2, 2)

    assert(all(size(src_pts) == size(dst_pts)), "src and dst points aren't the same size")
    assert(size(src_pts, 1) == 3 || size(src_pts, 2) == 3, "Expect 3xN or Nx3 matrix of points")

    % transpose to row major
    if size(src_pts, 1) == 3
        src_pts = src_pts';
        dst_pts = dst_pts';
    end

    % find mean/centroid
    centroid_src = mean(src_pts, 1);
    centroid_dst = mean(dst_pts, 1);

    % subtract mean
    src_pts = src_pts - centroid_src;
    dst_pts = dst_pts - centroid_dst;

    % this is similar to the cross-covariance matrix, except the outer mean is not calculated
    % https://en.wikipedia.org/wiki/Cross-covariance_matrix
    H = src_pts' * dst_pts;

    assert(size(H, 1) == 3 && size(H, 2) == 3, "H is not a 3x3 matrix")

    if rank(H) < 3
        error(sprintf("WARNING: rank of H = {%d}, expecting rank of 3 for a unique solution", rank(H)))
    end

    % find rotation
    [U,S,V] = svd(H);
    R = V*U';

    % special reflection case
    % https://en.wikipedia.org/wiki/Kabsch_algorithm
    if det(R) < 0
        printf("det(R) < 1, reflection detected!, correcting for it ...")
        R = V*diag([1, 1, -1])*U';
    end

    t = -R*centroid_src' + centroid_dst';
end
