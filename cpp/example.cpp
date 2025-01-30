#include <iostream>
#include <eigen3/Eigen/Geometry>
#include "rigid_transform_3D.hpp"

Eigen::MatrixXd random_points(int n) {
    Eigen::MatrixXd pts(n, 3);
    pts.setRandom();
    return pts;
}

Eigen::Matrix3d random_rotation() {
    Eigen::Vector4d q;
    q.setRandom();
    q.normalize();

    Eigen::Quaterniond R(q[0], q[1], q[2], q[3]);

    return R.matrix();
}

Eigen::Vector3d random_translation() {
    return Eigen::Vector3d::Random();
}

double random_scale() {
    while (true) {
        double s = std::abs(Eigen::Vector2d::Random()[0]) * 10.0;
        if (s > 0.1) {
            return s;
        }
    }
}

int main() {
    srand((unsigned int) time(0));

    Eigen::MatrixXd src_pts = random_points(100);
    Eigen::Matrix3d R = random_rotation();
    Eigen::Vector3d t = random_translation();
    double scale = random_scale();

    src_pts.setRandom();
    Eigen::MatrixXd dst_pts = scale * R * src_pts.transpose();
    dst_pts.transposeInPlace();

    for (int i=0; i < 3; i++) {
        dst_pts.col(i).array() += t(i);
    }

    auto [ret_R, ret_t, ret_scale] = rigid_transform_3D(src_pts, dst_pts, true);

    Eigen::MatrixXd dst_pts2 = ret_scale * ret_R * src_pts.transpose();
    dst_pts2.transposeInPlace();

    for (int i=0; i < 3; i++) {
        dst_pts2.col(i).array() += ret_t(i);
    }

    double rmse = std::sqrt((dst_pts - dst_pts2).array().square().mean());

    std::cout << "Ground truth rotation\n";
    std::cout << R << "\n";
    std::cout << "Recovered rotation\n";
    std::cout << ret_R << "\n\n";
    std::cout << "Ground truth translation: " << t.transpose() << "\n";
    std::cout << "Recovered translation: " << ret_t.transpose() << "\n\n";
    std::cout << "Ground truth scale: " << scale << "\n";
    std::cout << "Recovered scale: " << ret_scale << "\n\n";
    std::cout << "RMSE: " << rmse << "\n";

    if (rmse < 1e-5) {
        std::cout << "Everything looks good!\n";
    } else {
        std::cout << "Hmm something doesn't look right ...\n";
    }

    return 0;
}
