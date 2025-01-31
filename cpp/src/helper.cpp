#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <random>
#include <stdexcept>

Eigen::MatrixXd random_points(int n, int dim) {
    // returns points as row major
    Eigen::MatrixXd pts(n, dim);
    pts.setRandom();
    return pts;
}

Eigen::MatrixXd random_rotation(int dim) {
    Eigen::MatrixXd R;

    if (dim == 2) {
        std::default_random_engine gen;
        std::uniform_real_distribution<double> dice(-M_PI, M_PI);

        Eigen::Rotation2D rot(dice(gen));
        R = rot.matrix();
    } else if (dim == 3) {
        Eigen::Vector4d q;
        q.setRandom();
        q.normalize();

        Eigen::Quaterniond rot(q[0], q[1], q[2], q[3]);
        R = rot.matrix();
    } else {
        throw std::invalid_argument("dim must be 2 or 3");
    }

    return R;
}

Eigen::VectorXd random_translation(int dim) {
    Eigen::VectorXd t(dim);
    t.setRandom();
    return t;
}

double random_scale() {
    std::default_random_engine gen;
    std::uniform_real_distribution<double> dice(0.1, 10.0);
    return dice(gen);
}

Eigen::MatrixXd apply_transform(const Eigen::MatrixXd &pts,
                                const Eigen::MatrixXd &R,
                                const Eigen::VectorXd &t,
                                double scale) {
    // NOTE: pts are stored as rows major hence pts * R.transpose()
    Eigen::MatrixXd ret = scale * pts * R.transpose();

    int dim = static_cast<int>(pts.cols());

    for (int i = 0; i < dim; i++) {
        ret.col(i).array() += t(i);
    }

    return ret;
}
