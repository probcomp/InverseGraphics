import GeometryBasics: Point

function get_dims(box::S.Box)
    [box.sizeX, box.sizeY, box.sizeZ]
end

function xz_plane_aligned_bounding_box(point_cloud::Matrix, resolution::Real)
    cv2 = PyCall.pyimport("cv2")
    np = PyCall.pyimport("numpy")
    proj_onto_xz = GL.voxelize(vcat(point_cloud[1,:]', point_cloud[3, :]'), resolution)
    rect = cv2.minAreaRect(np.array(permutedims(proj_onto_xz), dtype="float32"))
    ((cx,cz), (width, height), rotation) = rect
    bottom, top = min_max(point_cloud[2, :])
    (bbox=S.Box(width, top-bottom, height), pose=Pose([cx, (top+bottom)/2, cz], R.RotY(-deg2rad(rotation))))
end

function axis_aligned_bounding_box(point_cloud::Matrix)
    mins, maxs = min_max(point_cloud[1:3,:]);
    (bbox=S.Box((maxs .- mins)...), pose=Pose([((mins .+ maxs) ./ 2.0)[:]...]))
end

function axis_aligned_bounding_box(point_clouds::Vector{Matrix{TYPE}}) where TYPE <: Real
    mins_maxs = min_max.(point_clouds)
    mins,_ = min_max(hcat([i[1][1:3] for i in mins_maxs]...))
    _, maxs = min_max(hcat([i[2][1:3] for i in mins_maxs]...))
    (bbox=S.Box((maxs .- mins)...), pose=Pose([((mins .+ maxs) ./ 2.0)[:]...]))
end

function get_boundary_mask(cloud::Matrix, bbox::S.Box, bbox_pose::Pose, wall_epsilon::Real, floor_epsilon::Real, ceiling_epsilon::Real)
    c = copy(cloud)
    c_centered = get_points_in_frame_b(c[1:3,:], bbox_pose)
    mask = fill(true, size(c_centered)[2])
    mask[c_centered[2,:] .> bbox.sizeY/2.0 - ceiling_epsilon] .= false
    mask[c_centered[2,:] .< -bbox.sizeY/2.0 + floor_epsilon] .= false
    mask[c_centered[1,:] .> bbox.sizeX/2.0 - wall_epsilon] .= false
    mask[c_centered[1,:] .< -bbox.sizeX/2.0 + wall_epsilon] .= false
    mask[c_centered[3,:] .> bbox.sizeZ/2.0 - wall_epsilon] .= false
    mask[c_centered[3,:] .< -bbox.sizeZ/2.0 + wall_epsilon] .= false
    mask
end

function subtract_boundary(cloud::Matrix, bbox::S.Box, bbox_pose::Pose, wall_epsilon::Real, floor_epsilon::Real, ceiling_epsilon::Real)
    c = copy(cloud)
    c[:, get_boundary_mask(cloud, bbox, bbox_pose, wall_epsilon, floor_epsilon, ceiling_epsilon)]
end


function get_bbox_corners(bbox::S.Box, pose::Pose)
    x,y,z = get_dims(bbox) ./ 2
    nominal_corners = collect([
        -x -y -z;
        -x y -z;
        x y -z;
        x -y -z;
        x y  z;
        x -y  z;
        -x -y  z;
        -x y  z;
    ]')
    corners = move_points_to_frame_b(nominal_corners, pose)
end

function get_bbox_corners_projected_to_xz(bbox::S.Box, pose::Pose)
    corners = get_bbox_corners(bbox,pose)
    corners[:, [1,3,5,7,1]]
end


function get_bbox_segments_point_list(bbox::S.Box, pose::Pose)
    x,y,z = get_dims(bbox) ./ 2
    nominal_corners = collect([
        # bottom square
        x y -z;
        x -y -z;

        x -y -z;
        -x -y -z;

        -x -y -z;
        -x y -z;

        -x y -z;
        x y -z;

        # top square
        x y z;
        x -y z;

        x -y z;
        -x -y z;

        -x -y z;
        -x y z;

        -x y z;
        x y z;

        # top to bottom
        -x -y z;
        -x -y -z;

        x -y z;
        x -y -z;

        -x y z;
        -x y -z;

        x y z;
        x y -z;
    ]')
    corners = move_points_to_frame_b(nominal_corners, pose)
    Point.(corners[1,:],corners[2,:], corners[3,:])
end
