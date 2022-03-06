function find_plane_inliers(cloud, plane_eq; threshold=0.1)
    inliers = abs.((plane_eq[1:3]' * cloud)[:] .+ plane_eq[4]) .< threshold
    mask = fill(false, size(cloud)[2])
    mask[inliers] .= true
    cloud[:, mask], mask
end

function find_plane(cloud; threshold= 0.1)
    pyrsc = PyCall.pyimport("pyransac3d")
    plane1 = pyrsc.Plane()
    best_eq, _ = plane1.fit(transpose(cloud), threshold)
    sub_cloud, inliers = find_plane_inliers(cloud, best_eq)
    best_eq, sub_cloud, inliers
end

function find_table_plane(cloud; max_iters=5)
    pyrsc = PyCall.pyimport("pyransac3d")
    for _ in 1:max_iters
        plane1 = pyrsc.Plane()
        best_eq, _ = plane1.fit(transpose(cloud), 0.1)

        if abs(best_eq[2]) > 0.5
            return best_eq
        else
            inliers = abs.((best_eq[1:3]' * cloud)[:] .+ best_eq[4]) .< 0.1
            mask = fill(false, size(cloud)[2])
            mask[inliers] .= true
            cloud = cloud[:, .!(mask)]
        end
    end
    return nothing
end

function camera_pose_from_table_eq(table_eq)
    a,b,c,d = table_eq
    if c > 0.0
        a,b,c,d = -1.0 .* [a,b,c,d]
    end
    shift = -d/c
    q = rotation_between_two_vectors([a,b,c], [0.0, 0.0, 1.0])
    camera_pose = inv(Pose([0.0, 0.0, shift], inv(q)))
end