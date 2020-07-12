#!/usr/bin/env python3

from numpy import *
from math import sqrt

from rigid_transform_3D import rigid_transform_3D

# Test with random data

# Random rotation and translation
R = mat(random.rand(3,3))
t = mat(random.rand(3,1))

# make R a proper rotation matrix, force orthonormal
U, S, Vt = linalg.svd(R)
R = U*Vt

# remove reflection
if linalg.det(R) < 0:
   Vt[2,:] *= -1
   R = U*Vt

# number of points
n = 10

# A = mat(random.rand(3, n));
A = array(random.rand(3, n));
B = R*A + tile(t, (1, n))

# Recover R and t
ret_R, ret_t = rigid_transform_3D(A, B)

# Compare the recovered R and t with the original
B2 = (ret_R*A) + tile(ret_t, (1, n))

# Find the root mean squared error
err = B2 - B
err = multiply(err, err)
err = sum(err)
rmse = sqrt(err/n);

print("Points A")
print(A)
print("")

print("Points B")
print(B)
print("")

print("Ground truth rotation")
print(R)

print("Recovered rotation")
print(ret_R)
print("")

print("Ground truth translation")
print(t)

print("Recovered translation")
print(ret_t)
print("")

print("RMSE:", rmse)

if rmse < 1e-5:
    print("Everything looks good!\n");
else:
    print("Hmm something doesn't look right ...\n");
