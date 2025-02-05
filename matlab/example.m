% Generate a random valid rotation matrix

function example()
    num_pts = 100;

    for dim = [2, 3]
        printf(sprintf("%dD points\n", dim));
        printf("")

        R = random_rotation(dim);
        t = random_translation(dim);
        scale = random_scale();
        src = random_points(num_pts, dim);
        dst = scale * src * R' + t';

        % Recover R and t
        [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);
        dst2 = ret_scale * src * R' + ret_t';

        rmse = sqrt(mean((dst - dst2).^2, "all"));

        printf("Ground truth rotation\n")
        R
        printf("\n")
        printf("Recovered rotation\n")
        ret_R
        printf("\n")

        printf("Ground truth translation: ")
        t'
        printf("Recovered translation: ")
        ret_t'
        printf("")
        printf("Ground truth scale: %f\n", scale)
        printf("Recovered scale: %f\n", ret_scale)
        printf("")
        printf("RMSE: %f", rmse)

        if rmse < 1e-6
            printf("Everything looks good!\n")
        else
            printf("Hmm something doesn't look right ...\n")
        end
    end
end

function pts = random_points(num, dim)
    pts = rand(num, dim)*2 - 1; % [-1, 1]
end

function R = random_rotation(dim)
    R = orth(rand(dim, dim)); % random rotation matrix

    if det(R) < 0
        [U, S, V] = svd(R);
        S = eye(dim);
        S(dim, dim) = -1;
        R = V * S * U';
    end
end

function t = random_translation(dim)
    t = rand(dim, 1);
end

function scale = random_scale()
    scale = 0.1 + rand() * 10;
end

