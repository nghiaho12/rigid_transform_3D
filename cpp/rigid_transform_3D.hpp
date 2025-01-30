#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/SVD>

struct RigidTransformResult {
    Eigen::MatrixXd R;
    Eigen::Vector3d t;
};

RigidTransformResult rigid_transform_3D(Eigen::MatrixXd src_pts, Eigen::MatrixXd dst_pts) {
    assert(src_pts.rows() == dst_pts.rows());
    assert(src_pts.cols() == dst_pts.cols());
    assert(src_pts.rows() == 3 || src_pts.cols() == 3);

    // transpose to row major
    if (src_pts.rows() == 3) {
        src_pts = src_pts.transpose();
        dst_pts = dst_pts.transpose();
    }
    std::cout << "HERE\n";

    // find mean/centroid
    Eigen::MatrixXd centroid_src(1, 3);
    Eigen::MatrixXd centroid_dst(1, 3);

    for (int i = 0; i < 3; i++) {
        centroid_src(0, i) = src_pts.col(i).mean();
        centroid_dst(0, i) = dst_pts.col(i).mean();
    }

    for (int i = 0; i < src_pts.rows(); i++) {
        src_pts.row(i) -= centroid_src.row(0);
        dst_pts.row(i) -= centroid_dst.row(0);
    }

    Eigen::MatrixXd H = src_pts.transpose() * dst_pts;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H);
    svd.compute(H, Eigen::ComputeThinV | Eigen::ComputeThinU);

    Eigen::MatrixXd R = svd.matrixV() * svd.matrixU().transpose();

    if (R.determinant() < 0) {
        Eigen::MatrixXd S(3, 3);

        S.setIdentity();
        S(2, 2) = -1.0;

        R = svd.matrixV() * S * svd.matrixU().transpose();
    }

    Eigen::Vector3d t = -R * centroid_src.transpose() + centroid_dst.transpose();

    return {R, t};
}
