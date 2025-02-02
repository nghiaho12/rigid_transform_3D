![C++](https://github.com/nghiaho12/rigid_transform_3D/actions/workflows/c-cpp.yml/badge.svg)
![Python](https://github.com/nghiaho12/rigid_transform_3D/actions/workflows/python-app.yml/badge.svg)

Code to find the rigid/Euclidean (rotation, translation) or similarity (rotation, translation, scale) transform between two sets of corresponding 2D/3D points.

# Usage
Code for C++, Python, Matlab/Octave can be found in their respective folder.

The function signature is
```
rigid_transform(src_pts, dst_pts, calc_scaling)
```

src_pts and dst_pts are points stored as rows in a matrix (e.g. Nx2 or Nx3).

# References
- https://en.wikipedia.org/wiki/Kabsch_algorithm
- "Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. and Huang, T. S. and Blostein, S. D, IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 9 Issue 5, May 1987.

