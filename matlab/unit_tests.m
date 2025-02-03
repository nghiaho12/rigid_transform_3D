function unit_tests()
    test_2d_not_enough_points();
    test_2d_min_points();
    test_2d_points();
    test_2d_rank_deficiency();
    test_2d_reflection();

    test_3d_not_enough_points();
    test_3d_min_points();
    test_3d_points();
    test_3d_rank_deficiency();
    test_3d_reflection();

    test_calc_scale_false();
    test_invalid_point_dim();
    test_mismatch_size();

    printf("All tests passed!\n")
end

function pts = random_points(num, dim)
    pts = rand(num, dim);
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

function ret = is_equal(A, B, tol=1e-6)
    diff = abs(A(:) - B(:));
    ret = all(diff < tol);
end

function test_2d_not_enough_points()
    src = [[1, 2]];

    caught = 0;
    try
        rigid_transform(src, src)
    catch
        caught = 1;
    end

    if ~caught
        error("expect assert/error thrown, but got none")
    end
end

function test_2d_min_points()
    dim = 2;

    R = random_rotation(dim);
    t = random_translation(dim);
    scale = random_scale();

    src = [1, 0; -1, 0];
    dst = scale * (src * R') + t';

    [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    assert(is_equal(R, ret_R))
    assert(is_equal(t, ret_t))
    assert(is_equal(scale, ret_scale))
end

function test_2d_points()
    dim = 2;
    num_pts = 100;

    R = random_rotation(dim);
    t = random_translation(dim);
    scale = random_scale();

    src = random_points(num_pts, dim);
    dst = scale * (src * R') + t';

    [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    assert(is_equal(R, ret_R))
    assert(is_equal(t, ret_t))
    assert(is_equal(scale, ret_scale))
end

function test_2d_rank_deficiency()
    src = [1, 1; 1, 1];

    caught = 0;
    try
        rigid_transform(src, src)
    catch
        caught = 1;
    end

    if ~caught
        error("expect assert/error thrown, but got none")
    end
end

function test_2d_reflection()
    R = [-0.63360525, 0.7736565; -0.7736565, -0.63360525];
    t = [-1; 1];
    scale = 1.23;

    src = [1, 0; -1, 0];
    dst = scale * (src * R') + t';

    [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    assert(is_equal(R, ret_R))
    assert(is_equal(t, ret_t))
    assert(is_equal(scale, ret_scale))
end

function test_3d_not_enough_points()
    src = [1, 2, 0; 3, 3, 0];

    caught = 0;
    try
        rigid_transform(src, src)
    catch
        caught = 1;
    end

    if ~caught
        error("expect assert/error thrown, but got none")
    end
end

function test_3d_min_points()
    dim = 3;

    R = random_rotation(dim);
    t = random_translation(dim);
    scale = random_scale();

    src = [1, 0, 0; -1, 0, 0; 1, 1, 0];
    dst = scale * (src * R') + t';

    [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    assert(is_equal(R, ret_R))
    assert(is_equal(t, ret_t))
    assert(is_equal(scale, ret_scale))
end

function test_3d_points()
    dim = 3;
    num_pts = 100;

    R = random_rotation(dim);
    t = random_translation(dim);
    scale = random_scale();

    src = random_points(num_pts, dim);
    dst = scale * (src * R') + t';

    [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    assert(is_equal(R, ret_R))
    assert(is_equal(t, ret_t))
    assert(is_equal(scale, ret_scale))
end

function test_3d_rank_deficiency()
    src = [0, 0, 0; 1, 1, 1; 2, 2, 2];

    caught = 0;
    try
        rigid_transform(src, src)
    catch
        caught = 1;
    end

    if ~caught
        error("expect assert/error thrown, but got none")
    end
end

function test_3d_reflection()
    R = [0.66962123, 0.22398235, 0.7081238;
        0.25805249, 0.8238756, -0.50461659;
        -0.69643113, 0.52063509, 0.49388539];
    t = [-1; 2; -3];
    scale = 2.34;

    src = [1, 0, 0; -1, 0, 0; 1, 1, 0];
    dst = scale * (src * R') + t';

    [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, true);

    assert(is_equal(R, ret_R))
    assert(is_equal(t, ret_t))
    assert(is_equal(scale, ret_scale))
end

function test_calc_scale_false()
    dim = 3;
    num_pts = 100;

    R = random_rotation(dim);
    t = random_translation(dim);
    scale = random_scale();

    src = random_points(num_pts, dim);
    dst = scale * (src * R') + t';

    [ret_R, ret_t, ret_scale] = rigid_transform(src, dst, false);

    assert(is_equal(R, ret_R))
    assert(is_equal(1.0, ret_scale))
end

function test_invalid_point_dim()
    dim = 4;
    num_pts = 100;

    src = random_points(num_pts, dim);

    caught = 0;
    try
        rigid_transform(src, src);
    catch
        caught = 1;
    end

    if ~caught
        error("expect assert/error thrown, but got none")
    end
end

function test_mismatch_size()
    num_pts = 100;

    src = random_points(num_pts, 2);
    dst = random_points(num_pts, 3);

    caught = 0;
    try
        rigid_transform(src, dst);
    catch
        caught = 1;
    end

    if ~caught
        error("expect assert/error thrown, but got none")
    end
end
