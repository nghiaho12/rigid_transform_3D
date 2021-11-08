% Generate a random valid rotation matrix
R = orth(rand(3,3)); % random rotation matrix

if det(R) < 0
    [U,S,V] = svd(R);
    V(:,3) = - V(:,3);
    R = V*U';
end

% Generate random translation
t = rand(3,1);

% Generate some random points
n = 10; % number of points
A = rand(3,n);

% Apply transform to get new dataset B
B = R*A + repmat(t, 1, n);

% Recover R and t
[ret_R, ret_t] = rigid_transform_3D(A, B);

% Compare the recovered R and t with the original
B2 = (ret_R*A) + repmat(ret_t, 1, n);

% Find the root mean squared error
err = B2 - B;
err = err .* err;
err = sum(err(:));
rmse = sqrt(err/n);

fprintf("Points A\n")
A

fprintf("Points B\n")
B

fprintf("Ground truth rotation\n")
R

fprintf("Recovered rotation\n")
ret_R

fprintf("Ground truth translation\n")
t

fprintf("Recovered translation\n")
ret_t

fprintf("RMSE: %f\n", rmse);

if rmse < 1e-5
    fprintf("Everything looks good!\n");
else
    fprintf("Hmm something doesn't look right ...\n");
end
