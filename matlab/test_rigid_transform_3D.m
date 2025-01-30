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
src_pts = rand(100, 3);

% Apply transform to get new dataset B
dst_pts = R*src_pts' + t;

% Recover R and t
[ret_R, ret_t] = rigid_transform_3D(src_pts, dst_pts');

dst_pts2 = ret_R * src_pts' + ret_t;
rmse = sqrt(mean((dst_pts - dst_pts2).^2, "all"));

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
