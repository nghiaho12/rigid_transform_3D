#include "rigid_transform.hpp"

#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>
#include <format>
#include <stdexcept>

RigidTransformResult rigid_transform(const Eigen::MatrixXd &src_pts, const Eigen::MatrixXd &dst_pts, bool calc_scale) {
    int dim = static_cast<int>(src_pts.cols());

    if (!(dim == 2 || dim == 3)) {
        throw std::invalid_argument("dim must be 2 or 3");
    }

    if (src_pts.rows() != dst_pts.rows()) {
        throw std::invalid_argument("src_pts.rows() != dst_pts.rows()");
    }

    if (src_pts.cols() != dst_pts.cols()) {
        throw std::invalid_argument("src_pts.cols() != dst_pts.cols()");
    }

    if (src_pts.rows() < dim) {
        throw std::invalid_argument(std::format("Not enough points, expect >= {}", dim));
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

    if (dim == 3 && svd.rank() < 2) {
        // For 3D points the rank can be 2, all points lie on a plane
        throw std::runtime_error(std::format("Insufficent matrix H rank, expect >= 2 but got {}, are the points collinear?", svd.rank()));
    } else if (dim == 2 && svd.rank() < 2) {
        throw std::runtime_error(std::format("Insufficent matrix H rank, expect 2 but got {}", svd.rank()));
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
