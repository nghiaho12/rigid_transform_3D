Implementation of the rigid 3D transform algorithm.

Given two sets of 3D points and their correspondence, the algorithm will return a least square optimal rigid transform (also known as Euclidean transform) between the two sets.
The transform solves for 3D rotation and 3D translation.

I added a flag to enable solving for scale as well. This will solve for the [similiarity transform](https://en.wikipedia.org/wiki/Similarity_(geometry). The scale is a single scalar.

# Usage
Go to the subfolder for the language you're interested and look at the test file for usage.

# References
- https://en.wikipedia.org/wiki/Kabsch_algorithm
- "Least-Squares Fitting of Two 3-D Point Sets", Arun, K. S. and Huang, T. S. and Blostein, S. D, IEEE Transactions on Pattern Analysis and Machine Intelligence, Volume 9 Issue 5, May 1987.

