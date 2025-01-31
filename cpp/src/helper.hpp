#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

Eigen::MatrixXd random_points(int n) {
    // returns points as row major
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

Eigen::Vector3d random_translation() { return Eigen::Vector3d::Random(); }

double random_scale() {
    while (true) {
        double s = std::abs(Eigen::Vector2d::Random()[0]) * 10.0;
        if (s > 0.1) {
            return s;
        }
    }
}

Eigen::MatrixXd apply_transform(const Eigen::MatrixXd &pts,
                                const Eigen::Matrix3d &R,
                                const Eigen::Vector3d &t,
                                double scale) {
    // NOTE: pts is row major hence pts * R.transpose()
    Eigen::MatrixXd ret = scale * pts * R.transpose();

    for (int i = 0; i < 3; i++) {
        ret.col(i).array() += t(i);
    }

    return ret;
}
