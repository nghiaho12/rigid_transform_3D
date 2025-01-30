#include <iostream>
#include "rigid_transform_3D.hpp"

int main() {
    Eigen::MatrixXd src_pts(100, 3);
    src_pts.setRandom();

    auto [R, t] = rigid_transform_3D(src_pts, src_pts);

    std::cout << R << "\n\n";
    std::cout << t << "\n\n";
    return 0;
}
