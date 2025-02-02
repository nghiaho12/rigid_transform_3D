Implementation of the rigid 3D transform algorithm.

Given two sets of points and their correspondences, the algorithm will find the optimal transform by solving a least squares problem.

2D and 3D points are supported as well as rigid (rotation, translation) and similiarity (rotation, translation, scale) transforms.

# Usage
Code for C++, Python, Matlab/Octave can be found in their respective folder.

The function signature is
```
rigid_transform(src_pts, dst_pts, calc_scaling)
```
and returns rotation, translation, scale.

src_pts and dst_pts are points stored as rows in a matrix (e.g. Nx2 or Nx3).

# References
- https://en.wikipedia.org/wiki/Kabsch_algorithm
- "Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. and Huang, T. S. and Blostein, S. D, IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 9 Issue 5, May 1987.

