import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import GenDirectionalStats as GDS
import NearestNeighbors
import LightGraphs as LG
import Gen
import PyCall
import PyCall: PyObject
import Statistics: mean
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end

V.setup_visualizer()

function CameraIntrinsics(step_metadata::PyCall.PyObject)::GL.CameraIntrinsics
    width, height = step_metadata.camera_aspect_ratio
    aspect_ratio = width / height

    # Camera principal point is the center of the image.
    cx, cy = width / 2.0, height / 2.0

    # Vertical field of view is given.
    fov_y = deg2rad(step_metadata.camera_field_of_view)
    # Convert field of view to distance to scale by aspect ratio and
    # convert back to radians to recover the horizontal field of view.
    fov_x = 2 * atan(aspect_ratio * tan(fov_y / 2.0))

    # Use the following relation to recover the focal length:
    #   FOV = 2 * atan( (0.5 * IMAGE_PLANE_SIZE) / FOCAL_LENGTH )
    fx = cx / tan(fov_x / 2.0)
    fy = cy / tan(fov_y / 2.0)

    clipping_near, clipping_far = step_metadata.camera_clipping_planes

    GL.CameraIntrinsics(width, height,
        fx, fy, cx, cy,
        clipping_near, clipping_far)
end
get_depth_image_from_step_metadata(step_metadata::PyObject) = Matrix(last(step_metadata.depth_map_list))
get_rgb_image_from_step_metadata(step_metadata::PyObject) = Int.(numpy.array(last(step_metadata.image_list)))

mcs = PyCall.pyimport("machine_common_sense")
controller = mcs.create_controller("../../../../GenPRAM_workspace/GenPRAM/assets/config_level1.ini")


scene_data, status = mcs.load_scene_json_file(
    "../../../../GenPRAM_workspace/GenPRAM/assets/eval_4_validation/eval_4_validation_collision_0001_03.json")
step_metadata_list = []
step_metadata = controller.start_scene(scene_data)
push!(step_metadata_list, step_metadata)
camera = CameraIntrinsics(step_metadata)

for _ in 1:150
step_metadata = controller.step("Pass")
push!(step_metadata_list, step_metadata)
end

function subtract_floor_and_back_wall(c, threhsold)
    _, floor_y, back_wall_z = maximum(c;dims=2)[:]
    c = c[:,c[2,:] .< (floor_y - threhsold)]
    c = c[:,c[3,:] .< (back_wall_z - threhsold)]
    c
end
voxelize_resolution = 0.05
depth_images = get_depth_image_from_step_metadata.(step_metadata_list)
obs_clouds = map(x->GL.depth_image_to_point_cloud(x,camera), depth_images);
obs_clouds = map(x->subtract_floor_and_back_wall(x,0.1), obs_clouds);
obs_clouds = T.voxelize.(obs_clouds, voxelize_resolution)
entities_list = [T.get_entities_from_assignment(c1,T.dbscan_cluster(c1)) for c1 in obs_clouds];

length.(entities_list)

t = 145
c1 = entities_list[t][1]
c2 = entities_list[t+2][1]
V.reset_visualizer()
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(c2 ./ 10.0; color=I.colorant"black", channel_name=:h2)


c1 = T.voxelize(c1, 0.1)
c2 = T.voxelize(c2, 0.1)
@show size(c1), size(c2)
V.reset_visualizer()
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(c2 ./ 10.0; color=I.colorant"black", channel_name=:h2)

import PyCall
np = PyCall.pyimport("numpy")
py_goicp = PyCall.pyimport("py_goicp")

# +
function normalize_cloud(cloud)
    mi = minimum(cloud;dims=2)
    ma = maximum(cloud;dims=2)
    range = (ma .- mi)
    range[abs.(range) .< 0.001] .= 1.0
    cloud = (cloud .- mi) ./ range
end
function prep_cloud(cloud)
    p3dlist = [];
    for (x,y,z) in eachcol(cloud)
        pt = py_goicp.POINT3D(x,y,z);
        push!(p3dlist,pt)
    end
    return length(p3dlist), p3dlist
end

Nm, a_points = prep_cloud(c1)
Nd, b_points = prep_cloud(c2)

# -

goicp = py_goicp.GoICP();
goicp.loadModelAndData(Nm, a_points, Nd, b_points);
goicp.setDTSizeAndFactor(10, 1.0);
goicp.BuildDT();
goicp.Register();

goicp.optimalTranslation()

# +

rot = R.RotMatrix{3}(goicp.optimalRotation()); # A python list of 3x3 is returned with the optimal rotation
translation  = goicp.optimalTranslation()# A python list of 1x3 is returned with the optimal translation
offset_pose = Pose(translation, rot);
V.reset_visualizer()
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(T.move_points_to_frame_b(c2, offset_pose) ./ 10.0; color=I.colorant"black", channel_name=:h2)
# -





# +

centroid = mean(c1;dims=2) .- mean(c2;dims=2)
shift = Pose(centroid[:])
c2 = T.move_points_to_frame_b(c2, shift)

p = T.icp(c1,c2)
c2 = T.move_points_to_frame_b(c2, p)

V.viz(c2 ./ 10.0; color=I.colorant"black", channel_name=:h2)
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
# -

V.reset_visualizer()
V.viz(c2 ./ 10.0; color=I.colorant"black", channel_name=:h2)
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)

V.reset_visualizer()
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(c2 ./ 10.0; color=I.colorant"black", channel_name=:h2)


random_pose = Pose([Gen.uniform(-0.05, 0.05) for _ in 1:3],GDS.vmf_rot3(IDENTITY_ORN, 5000.0))
c2 = T.move_points_to_frame_b(c2, random_pose)
p = T.icp(c1,c2)
c2 = T.move_points_to_frame_b(c2, p)






V.reset_visualizer()
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(c2 ./ 10.0; color=I.colorant"black", channel_name=:h2)

c2 = T.move_points_to_frame_b
centroid = mean(c1;dims=2) .- mean(c2;dims=2)
shift = Pose(centroid[:])
c2 = T.move_points_to_frame_b(c2, shift)
V.reset_visualizer()
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(c2 ./ 10.0; color=I.colorant"black", channel_name=:h2)





meshes = map(x -> GL.mesh_from_voxelized_cloud(x, 0.1), entities);

scaled_camera = T.scale_down_camera(camera, 8)
renderer = GL.setup_renderer(scaled_camera, GL.DepthMode())
resolution = 0.01
for m in meshes
    GL.load_object!(renderer, m)
end
# Helper function to get point cloud from the object ids, object poses, and camera pose
function get_cloud(poses, ids, camera_pose)
    depth_image = GL.gl_render(renderer, ids, poses, camera_pose)
    cloud = GL.depth_image_to_point_cloud(depth_image, scaled_camera)
    if isnothing(cloud)
        cloud = zeros(3,1)
    end
    cloud
end

depth_image = GL.gl_render(renderer, [2,1], [IDENTITY_POSE,IDENTITY_POSE], IDENTITY_POSE)
GL.view_depth_image(depth_image)

rerendered_cloud = get_cloud([IDENTITY_POSE,IDENTITY_POSE], [1,2], IDENTITY_POSE);
V.reset_visualizer()
V.viz(rerendered_cloud ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(c2 ./ 10.0; color=I.colorant"black", channel_name=:h2)


T.axis_aligned_bbox_from_point_cloud(entities[1])


# +
refined_pose1 = T.icp_object_pose(
    IDENTITY_POSE,
    c2,
    p -> T.voxelize(get_cloud([p], [1], IDENTITY_POSE), 0.05)
)

refined_pose2 = T.icp_object_pose(
    IDENTITY_POSE,
    c2,
    p -> T.voxelize(get_cloud([p], [2], IDENTITY_POSE), 0.05)
)

# -



@time rerendered_cloud = T.voxelize(get_cloud([refined_pose1,refined_pose2], [1,2], IDENTITY_POSE), 0.05);
V.reset_visualizer()
V.viz(rerendered_cloud ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(c2 ./ 10.0; color=I.colorant"black", channel_name=:h2)


T.voxelize






