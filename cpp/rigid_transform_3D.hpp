#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

struct RigidTransformResult {
    Eigen::MatrixXd R;
    Eigen::Vector3d t;
    double scale = 1.0;
};

RigidTransformResult rigid_transform_3D(Eigen::MatrixXd src_pts, Eigen::MatrixXd dst_pts, bool calc_scale=false) {
    assert(src_pts.rows() == dst_pts.rows());
    assert(src_pts.cols() == dst_pts.cols());
    assert(src_pts.rows() == 3 || src_pts.cols() == 3); // invalid matrix
    assert(std::min(src_pts.rows(), src_pts.cols()) >= 3); // not enough points

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

    Eigen::MatrixXd R = svd.matrixV() * svd.matrixU().transpose();

    if (R.determinant() < 0) {
        Eigen::MatrixXd S(3, 3);

        S.setIdentity();
        S(2, 2) = -1.0;

        R = svd.matrixV() * S * svd.matrixU().transpose();
    }

    double scale = 1.0;

    if (calc_scale) {
        scale = std::sqrt(dst_pts.array().square().sum() / src_pts.array().square().sum());
    }

    Eigen::Vector3d t = -scale * (R * centroid_src) + centroid_dst;

    return {R, t, scale};
}
