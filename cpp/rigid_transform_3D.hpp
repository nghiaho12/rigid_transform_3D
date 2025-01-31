#include <stdexcept>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

struct RigidTransformResult {
    Eigen::MatrixXd R;
    Eigen::Vector3d t;
    double scale = 1.0;
};

RigidTransformResult rigid_transform_3D(Eigen::MatrixXd src_pts, Eigen::MatrixXd dst_pts, bool calc_scale=false) {
    if (src_pts.rows() != dst_pts.rows()) {
        throw std::invalid_argument("src_pts.rows() != dst_pts.rows()");
    }

    if (src_pts.cols() != dst_pts.cols()) {
        throw std::invalid_argument("src_pts.cols() != dst_pts.cols()");
    }

    if (src_pts.rows() != 3 && src_pts.cols() != 3) {
        throw std::invalid_argument("points are not 3D");
    }

    if (std::min(src_pts.rows(), src_pts.cols()) < 3) {
        throw std::invalid_argument("expect >= 3 points");
    }

    // transpose to row major
    if (src_pts.rows() == 3) {
        src_pts.transposeInPlace();
        dst_pts.transposeInPlace();
    }

    // find mean/centroid
    Eigen::Vector3d centroid_src;
    Eigen::Vector3d centroid_dst;

    for (int i = 0; i < 3; i++) {
        centroid_src(i) = src_pts.col(i).mean();
        centroid_dst(i) = dst_pts.col(i).mean();
    }

    for (int i = 0; i < 3; i++) {
        src_pts.col(i).array() -= centroid_src(i);
        dst_pts.col(i).array() -= centroid_dst(i);
    }

    Eigen::MatrixXd H = src_pts.transpose() * dst_pts;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H);
    svd.compute(H, Eigen::ComputeFullV | Eigen::ComputeFullU);

    if (svd.rank() < 3) {
        throw std::runtime_error("rank of H is < 3, no unique solution");
    }

    Eigen::MatrixXd R = svd.matrixV() * svd.matrixU().transpose();

    if (R.determinant() < 0) {
        Eigen::MatrixXd S(3, 3);

        S.setIdentity();
        S(2, 2) = -1.0;

        R = svd.matrixV() * S * svd.matrixU().transpose();
    }

    double scale = 1.0;

    if (calc_scale) {
        double num = dst_pts.array().square().sum();
        double den = src_pts.array().square().sum();

        if (den < 1e-6) {
            throw std::runtime_error("division by zero when calculating scale");
        }

        scale = std::sqrt(num / den);
    }

    Eigen::Vector3d t = -scale * R * centroid_src + centroid_dst;

    return {R, t, scale};
}
