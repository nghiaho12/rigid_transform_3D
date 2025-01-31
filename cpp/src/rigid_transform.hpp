#include <eigen3/Eigen/Dense>

struct RigidTransformResult {
    Eigen::MatrixXd R;
    Eigen::VectorXd t;
    double scale = 1.0;
};

// Calculates the optimal rigid transform from src_pts to dst_pts.
//
// The returned transform minimizes the following least-squares problem
//     r = dst_pts - (R * src_pts + t)
//     s = sum(r**2))
//
// If calc_scale is True, the similarity transform is solved, with the residual being
//     r = dst_pts - (scale * R * src_pts + t)
// where scale is a scalar.
//
// Parameters
// ----------
// src_pts: matrix of points stored as rows (e.g. Nx3)
// dst_pts: matrix of points stored as rows (e.g. Nx3)
// calc_scale: if true solve for scale
RigidTransformResult rigid_transform(const Eigen::MatrixXd &src_pts,
                                     const Eigen::MatrixXd &dst_pts,
                                     bool calc_scale = false);
