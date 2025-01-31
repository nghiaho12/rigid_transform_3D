#include <time.h>

#include <format>
#include <iostream>

#include "helper.hpp"
#include "rigid_transform.hpp"

int main() {
    srand(static_cast<unsigned int>(time(0)));

    for (int dim : {2, 3}) {
        std::cout << std::format("{:=^60}\n", "");
        std::cout << std::format("{}D points\n\n", dim);

        Eigen::MatrixXd src_pts = random_points(100, dim);
        Eigen::MatrixXd R = random_rotation(dim);
        Eigen::VectorXd t = random_translation(dim);
        double scale = random_scale();

        Eigen::MatrixXd dst_pts = apply_transform(src_pts, R, t, scale);

        // recover the transform
        auto [ret_R, ret_t, ret_scale] = rigid_transform(src_pts, dst_pts, true);
        Eigen::MatrixXd dst_pts2 = apply_transform(src_pts, ret_R, ret_t, ret_scale);

        // this should be close to zero
        double rmse = std::sqrt((dst_pts - dst_pts2).array().square().mean());

        std::cout << "Ground truth rotation\n";
        std::cout << R << "\n\n";
        std::cout << "Recovered rotation\n";
        std::cout << ret_R << "\n\n";
        std::cout << "Ground truth translation: " << t.transpose() << "\n";
        std::cout << "Recovered translation: " << ret_t.transpose() << "\n\n";
        std::cout << "Ground truth scale: " << scale << "\n";
        std::cout << "Recovered scale: " << ret_scale << "\n\n";
        std::cout << "RMSE: " << rmse << "\n";

        if (rmse < 1e-6) {
            std::cout << "Everything looks good!\n";
        } else {
            std::cout << "Hmm something doesn't look right ...\n";
        }
    }

    return 0;
}
