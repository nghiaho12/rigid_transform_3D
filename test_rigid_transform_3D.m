% Generate a random valid rotation matrix
R = orth(rand(3,3)); % random rotation matrix

if det(R) < 0
    [U,S,V] = svd(R);
    V(:,3) *= -1;
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

printf("Points A\n")
A

printf("Points B\n")
B

printf("Ground truth rotation\n")
R

printf("Recovered rotation\n")
ret_R

printf("Ground truth translation\n")
t

printf("Recovered translation\n")
ret_t

printf("RMSE: %f\n", rmse);

if rmse < 1e-5
    printf("Everything looks good!\n");
else
    printf("Hmm something doesn't look right ...\n");
end
