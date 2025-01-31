#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "helper.hpp"
#include "rigid_transform.hpp"

constexpr double TOL = 1e-6;

TEST_CASE("2d points") {
    int dim = 2;
    Eigen::MatrixXd src = random_points(100, dim);
    Eigen::MatrixXd R = random_rotation(dim);
    Eigen::VectorXd t = random_translation(dim);
    double scale = random_scale();

    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    REQUIRE(R.isApprox(ret_R));
    REQUIRE(t.isApprox(ret_t));
    REQUIRE_THAT(scale, Catch::Matchers::WithinAbs(ret_scale, TOL));
}

TEST_CASE("3d points") {
    int dim = 3;
    Eigen::MatrixXd src = random_points(100, dim);
    Eigen::MatrixXd R = random_rotation(dim);
    Eigen::VectorXd t = random_translation(dim);
    double scale = random_scale();

    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    REQUIRE(R.isApprox(ret_R));
    REQUIRE(t.isApprox(ret_t));
    REQUIRE_THAT(scale, Catch::Matchers::WithinAbs(ret_scale, TOL));
}

TEST_CASE("calc_scale false") {
    int dim = 3;
    Eigen::MatrixXd src = random_points(100, dim);
    Eigen::MatrixXd R = random_rotation(dim);
    Eigen::VectorXd t = random_translation(dim);
    double scale = 10.0;

    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, false);

    REQUIRE(R.isApprox(ret_R));
    REQUIRE_THAT(ret_scale, Catch::Matchers::WithinAbs(1.0, TOL));
}

TEST_CASE("invalid point dim") {
    int dim = 4;
    Eigen::MatrixXd src = random_points(100, dim);
    REQUIRE_THROWS(rigid_transform(src, src));
}

TEST_CASE("mismatch in matrix size") {
    Eigen::MatrixXd src = random_points(100, 2);
    Eigen::MatrixXd dst = random_points(100, 3);
    REQUIRE_THROWS(rigid_transform(src, dst));
}

TEST_CASE("wrong order") {
    Eigen::MatrixXd src = random_points(3, 100);
    REQUIRE_THROWS(rigid_transform(src, src));
}

TEST_CASE("not enough points") {
    int dim = 3;
    Eigen::MatrixXd src = random_points(2, dim);
    REQUIRE_THROWS(rigid_transform(src, src));
}

TEST_CASE("2d all identical points") {
    int dim = 2;
    Eigen::MatrixXd src(3, dim);
    src.setOnes();

    REQUIRE_THROWS(rigid_transform(src, src));
}

TEST_CASE("3d collinear points") {
    int dim = 3;
    Eigen::MatrixXd src(4, dim);
    src << 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3;

    REQUIRE_THROWS(rigid_transform(src, src));
}

TEST_CASE("3d points on a 2d plane") {
    int dim = 3;
    Eigen::MatrixXd src = random_points(100, dim);
    Eigen::MatrixXd R = random_rotation(dim);
    Eigen::VectorXd t = random_translation(dim);
    double scale = random_scale();

    src.col(2) *= 0.0;
    Eigen::MatrixXd dst = apply_transform(src, R, t, scale);

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);
    REQUIRE(R.isApprox(ret_R));
    REQUIRE(t.isApprox(ret_t));
    REQUIRE_THAT(scale, Catch::Matchers::WithinAbs(ret_scale, TOL));
}

TEST_CASE("reflection") {
    int dim = 2;
    Eigen::MatrixXd src(3, dim);
    Eigen::MatrixXd dst(3, dim);

    src << 0.04997603, 0.92769423, 0.64045334, 0.5110098, 0.80452329, 0.19618526;

    dst << 0.44224556, 0.61291874, 0.4559217, 0.17549108, 0.57923716, 0.23585888;

    auto [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, false);

    REQUIRE_THAT(ret_R.determinant(), Catch::Matchers::WithinAbs(1.0, TOL));
}
