#include "rigid_transform.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <format>

RigidTransformResult rigid_transform(const Eigen::MatrixXd &src_pts, const Eigen::MatrixXd &dst_pts, bool calc_scale) {
    int dim = static_cast<int>(src_pts.cols());

    if (src_pts.rows() != dst_pts.rows() || src_pts.cols() != dst_pts.cols()) {
        throw SrcDstSizeMismatchError(
            std::format("src and dst points aren't the same matrix size {}x{} != {}x{}", src_pts.rows(), src_pts.cols(), dst_pts.rows(), dst_pts.cols()));
    }

    if (!(dim == 2 || dim == 3)) {
        throw InvalidPointDimError(std::format("Points must be 2D or 3D, src_pts.shape[1] = {}", dim));
    }

    if (src_pts.rows() < dim) {
        throw NotEnoughPointsError(std::format("Not enough points, expect >= {} points", dim));
    }

    // find mean/centroid
    Eigen::VectorXd centroid_src(dim);
    Eigen::VectorXd centroid_dst(dim);

    for (int i = 0; i < dim; i++) {
        centroid_src(i) = src_pts.col(i).mean();
        centroid_dst(i) = dst_pts.col(i).mean();
    }

    int n = static_cast<int>(src_pts.rows());
    Eigen::MatrixXd src_pts2 = src_pts - centroid_src.transpose().replicate(n, 1);
    Eigen::MatrixXd dst_pts2 = dst_pts - centroid_dst.transpose().replicate(n, 1);
    Eigen::MatrixXd H = src_pts2.transpose() * dst_pts2;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H);
    svd.compute(H, Eigen::ComputeFullV | Eigen::ComputeFullU);

    auto rank = svd.rank();
    if (dim == 2 && rank == 0) {
        throw RankDeficiencyError(
            std::format("Insufficent matrix rank. For 2D points expect rank >= 1 but got {}. Maybe your points are all the same?", rank));
    } else if (dim == 3 && rank <= 1) {
        throw RankDeficiencyError(
            std::format("Insufficent matrix rank. For 3D points expect rank >= 2 but got {}. Maybe your points are collinear?", rank));
    }

    Eigen::MatrixXd R = svd.matrixV() * svd.matrixU().transpose();

    if (R.determinant() < 0) {
        Eigen::MatrixXd S(dim, dim);

        S.setIdentity();
        S(dim - 1, dim - 1) = -1.0;

        R = svd.matrixV() * S * svd.matrixU().transpose();
    }

    double scale = 1.0;

    if (calc_scale) {
        double num = dst_pts2.array().square().mean();
        double den = src_pts2.array().square().mean();

        scale = std::sqrt(num / den);
    }

    Eigen::VectorXd t = -scale * R * centroid_src + centroid_dst;

    return {R, t, scale};
}
