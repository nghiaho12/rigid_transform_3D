#include <catch2/catch_test_macros.hpp>

#include "helper.hpp"
#include "rigid_transform_3D.hpp"

constexpr double TOL = 1e-6;

TEST_CASE("example") {
    Eigen::MatrixXd src_pts = random_points(100);
    Eigen::Matrix3d R = random_rotation();
    Eigen::Vector3d t = random_translation();
    double scale = random_scale();

    Eigen::MatrixXd dst_pts = apply_transform(src_pts, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform_3D(src_pts, dst_pts, true);

    REQUIRE((R - ret_R).sum() < TOL);
    REQUIRE((t - ret_t).sum() < TOL);
    REQUIRE(std::abs(scale - ret_scale) < TOL);
}

TEST_CASE("identity transform") {
    Eigen::MatrixXd pts = random_points(100);
    auto [R, t, scale] = rigid_transform_3D(pts, pts);

    REQUIRE((R - Eigen::Matrix3d::Identity()).sum() < TOL);
    REQUIRE(std::abs(t.sum()) < TOL);
    REQUIRE(std::abs(scale - 1.0) < TOL);
}

TEST_CASE("mismatch in matrix size") {
    Eigen::MatrixXd src_pts = random_points(100);
    Eigen::MatrixXd dst_pts = random_points(10);
    REQUIRE_THROWS(rigid_transform_3D(src_pts, dst_pts));
}

TEST_CASE("not 3D points") {
    Eigen::MatrixXd src_pts(100, 2);
    REQUIRE_THROWS(rigid_transform_3D(src_pts, src_pts));
}

TEST_CASE("not enough points") {
    Eigen::MatrixXd pts = random_points(2);
    REQUIRE_THROWS(rigid_transform_3D(pts, pts));
}

TEST_CASE("low rank matrix") {
    Eigen::MatrixXd pts(100, 3);
    pts.setZero();
    REQUIRE_THROWS(rigid_transform_3D(pts, pts));
}
