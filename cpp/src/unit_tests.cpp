#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "helper.hpp"
#include "rigid_transform.hpp"

constexpr double TOL = 1e-6;

TEST_CASE("2d not enough points") {
    Eigen::MatrixXd src(1, 2);
    REQUIRE_THROWS_AS(rigid_transform(src, src), NotEnoughPointsError);
}

TEST_CASE("2d min points") {
    int dim = 2;
    Eigen::MatrixXd R = random_rotation(dim);
    Eigen::VectorXd t = random_translation(dim);
    double scale = random_scale();

    Eigen::MatrixXd src(2, dim);
    src << 1, 0, -1, 0;
    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    REQUIRE(R.isApprox(ret_R, TOL));
    REQUIRE(t.isApprox(ret_t, TOL));
    REQUIRE_THAT(scale, Catch::Matchers::WithinAbs(ret_scale, TOL));
}

TEST_CASE("2d points") {
    int dim = 2;
    int num_pts = 100;

    Eigen::MatrixXd src = random_points(num_pts, dim);
    Eigen::MatrixXd R = random_rotation(dim);
    Eigen::VectorXd t = random_translation(dim);
    double scale = random_scale();

    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    REQUIRE(R.isApprox(ret_R, TOL));
    REQUIRE(t.isApprox(ret_t, TOL));
    REQUIRE_THAT(scale, Catch::Matchers::WithinAbs(ret_scale, TOL));
}

TEST_CASE("2d rank deficiency") {
    Eigen::MatrixXd src = random_points(2, 2);
    src << 1, 1, 1, 1;
    REQUIRE_THROWS_AS(rigid_transform(src, src), RankDeficiencyError);
}

TEST_CASE("2d reflection") {
    Eigen::Matrix2d R;
    Eigen::Vector2d t;
    Eigen::MatrixXd src(2, 2);

    R << -0.63360525, 0.7736565, -0.7736565, -0.63360525;
    t << -1, 1;
    double scale = 1.23;

    src << 1, 0, -1, 0;

    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    REQUIRE(R.isApprox(ret_R, TOL));
    REQUIRE(t.isApprox(ret_t, TOL));
    REQUIRE_THAT(scale, Catch::Matchers::WithinAbs(ret_scale, TOL));
}

TEST_CASE("3d not enough points") {
    Eigen::MatrixXd src(2, 3);
    REQUIRE_THROWS_AS(rigid_transform(src, src), NotEnoughPointsError);
}

TEST_CASE("3d min points") {
    int dim = 3;
    Eigen::MatrixXd R = random_rotation(dim);
    Eigen::VectorXd t = random_translation(dim);
    double scale = random_scale();

    Eigen::MatrixXd src(3, dim);
    src << 1, 0, 0, -1, 0, 0, 1, 1, 0;
    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    REQUIRE(R.isApprox(ret_R, TOL));
    REQUIRE(t.isApprox(ret_t, TOL));
    REQUIRE_THAT(scale, Catch::Matchers::WithinAbs(ret_scale, TOL));
}

TEST_CASE("3d points") {
    int dim = 3;
    int num_pts = 100;

    Eigen::MatrixXd src = random_points(num_pts, dim);
    Eigen::MatrixXd R = random_rotation(dim);
    Eigen::VectorXd t = random_translation(dim);
    double scale = random_scale();

    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    REQUIRE(R.isApprox(ret_R, TOL));
    REQUIRE(t.isApprox(ret_t, TOL));
    REQUIRE_THAT(scale, Catch::Matchers::WithinAbs(ret_scale, TOL));
}

TEST_CASE("3d rank deficiency") {
    int dim = 3;
    int num_pts = 3;
    Eigen::MatrixXd src(num_pts, dim);
    src << 0, 0, 0, 1, 1, 1, 2, 2, 2;
    REQUIRE_THROWS_AS(rigid_transform(src, src), RankDeficiencyError);
}

TEST_CASE("3d reflection") {
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::MatrixXd src(3, 3);

    R << 0.66962123, 0.22398235, 0.7081238,
                0.25805249, 0.8238756, -0.50461659,
                -0.69643113, 0.52063509, 0.49388539;
    t << -1, 2, -3;
    double scale = 2.34;

    src << 1, 0, 0, -1, 0, 0, 1, 1, 0;

    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    REQUIRE(R.isApprox(ret_R, TOL));
    REQUIRE(t.isApprox(ret_t, TOL));
    REQUIRE_THAT(scale, Catch::Matchers::WithinAbs(ret_scale, TOL));
}

TEST_CASE("calc scale false") {
    int dim = 3;
    int num_pts = 100;

    Eigen::MatrixXd src = random_points(num_pts, dim);
    Eigen::MatrixXd R = random_rotation(dim);
    Eigen::VectorXd t = random_translation(dim);
    double scale = random_scale();

    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, false);

    REQUIRE(R.isApprox(ret_R, TOL));
    REQUIRE_THAT(1.0, Catch::Matchers::WithinAbs(ret_scale, TOL));
}

TEST_CASE("invalid point dim") {
    int dim = 4;
    int num_pts = 100;

    Eigen::MatrixXd src = random_points(num_pts, dim);

    REQUIRE_THROWS_AS(rigid_transform(src, src), InvalidPointDimError);
}

TEST_CASE("mismatch size") {
    int num_pts = 100;

    Eigen::MatrixXd src = random_points(num_pts, 2);
    Eigen::MatrixXd dst= random_points(num_pts, 3);

    REQUIRE_THROWS_AS(rigid_transform(src, dst), SrcDstSizeMismatchError);
}
