#include <eigen3/Eigen/Dense>

Eigen::MatrixXd random_points(int n, int dim);
Eigen::MatrixXd random_rotation(int dim);
Eigen::VectorXd random_translation(int dim);
double random_scale();

Eigen::MatrixXd apply_transform(const Eigen::MatrixXd &pts,
                                const Eigen::MatrixXd &R,
                                const Eigen::VectorXd &t,
                                double scale);
