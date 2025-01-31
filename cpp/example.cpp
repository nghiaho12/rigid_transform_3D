#include <time.h>
#include <iostream>
#include "helper.hpp"
#include "rigid_transform_3D.hpp"

int main() {
    // generate some random points and transform
    srand(static_cast<unsigned int>(time(0)));

    Eigen::MatrixXd src_pts = random_points(100);
    Eigen::Matrix3d R = random_rotation();
    Eigen::Vector3d t = random_translation();
    double scale = random_scale();

    Eigen::MatrixXd dst_pts = apply_transform(src_pts, R, t, scale);

    // recover the transform
    auto [ret_R, ret_t, ret_scale] = rigid_transform_3D(src_pts, dst_pts, true);
    Eigen::MatrixXd dst_pts2 = apply_transform(src_pts, ret_R, ret_t, ret_scale);

    // this should be close to zero
    double rmse = std::sqrt((dst_pts - dst_pts2).array().square().mean());

    std::cout << "Ground truth rotation\n";
    std::cout << R << "\n\n1";
    std::cout << "Recovered rotation\n";
    std::cout << ret_R << "\n\n";
    std::cout << "Ground truth translation: " << t.transpose() << "\n";
    std::cout << "Recovered translation: " << ret_t.transpose() << "\n\n";
    std::cout << "Ground truth scale: " << scale << "\n";
    std::cout << "Recovered scale: " << ret_scale << "\n\n";
    std::cout << "RMSE: " << rmse << "\n";

    return 0;
}
