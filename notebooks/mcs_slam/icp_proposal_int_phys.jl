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
import FileIO
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
numpy = PyCall.pyimport("numpy")
get_depth_image_from_step_metadata(step_metadata::PyObject) = Matrix(last(step_metadata.depth_map_list))
get_rgb_image_from_step_metadata(step_metadata::PyObject) = Float64.(numpy.array(last(step_metadata.image_list)))

mcs = PyCall.pyimport("machine_common_sense")
controller = mcs.create_controller("../../../../GenPRAM_workspace/GenPRAM/assets/config_level2.ini")


scene_data, status = mcs.load_scene_json_file(
#     "../../../../GenPRAM_workspace/GenPRAM/assets/eval_4_all/november_0001_01.json")
    "../../../../GenPRAM_workspace/GenPRAM/assets/eval_4_all/quebec_0002_01.json")
step_metadata_list = []
step_metadata = controller.start_scene(scene_data)
push!(step_metadata_list, step_metadata)
camera = CameraIntrinsics(step_metadata)

for _ in 1:150
    step_metadata = controller.step("Pass")
    if isnothing(step_metadata)
        break
    end
    push!(step_metadata_list, step_metadata)
end

function subtract_floor_and_back_wall(c, threhsold)
    _, floor_y, back_wall_z = maximum(c;dims=2)[:]
    c = c[:,c[2,:] .< (floor_y - threhsold)]
    c = c[:,c[3,:] .< (back_wall_z - threhsold)]
    c
end
voxelize_resolution = 0.05
rgb_images = get_rgb_image_from_step_metadata.(step_metadata_list)
depth_images = get_depth_image_from_step_metadata.(step_metadata_list)
obs_clouds = map(x->GL.depth_image_to_point_cloud(x,camera), depth_images);
obs_clouds = map(x->subtract_floor_and_back_wall(x,0.1), obs_clouds);
obs_clouds = T.voxelize.(obs_clouds, voxelize_resolution)
entities_list = [T.get_entities_from_assignment(c1,T.dbscan_cluster(c1)) for c1 in obs_clouds];

# +
timesteps = collect(96:100);
t = timesteps[1]
# t = 90
V.reset_visualizer()

V.viz(obs_clouds[t] ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(obs_clouds[t+1] ./ 10.0; color=I.colorant"black", channel_name=:h2)

GL.view_rgb_image(rgb_images[t];in_255=true)

# +
t = timesteps[end]
# t = 90
V.reset_visualizer()

V.viz(obs_clouds[t] ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(obs_clouds[t+1] ./ 10.0; color=I.colorant"black", channel_name=:h2)

GL.view_rgb_image(rgb_images[t];in_255=true)
# -

score_clouds(obs_cloud, gen_cloud) = Gen.logpdf(T.uniform_mixture_from_template, obs_cloud, gen_cloud, 0.0001, 3*voxelize_resolution, (-100.0, 100.0, -100.0, 100.0,-100.0,100.0))
@show score_clouds(obs_clouds[t+1], obs_clouds[t])
@show score_clouds(obs_clouds[t+1], obs_clouds[t+1])

# +
cam = GL.scale_down_camera(camera, 4)
renderer = GL.setup_renderer(cam, GL.DepthMode());
t = timesteps[1]
e = entities_list[t]
@show length(e)
start_poses = (x -> Pose(x)).(T.centroid.(e))
centered_entities = [x .- c.pos for (x,c) in zip(e, start_poses)]
meshes = [
    GL.mesh_from_voxelized_cloud(x, voxelize_resolution)
    for x in centered_entities
];

for m in meshes
GL.load_object!(renderer, m)
end
d = GL.gl_render(renderer, collect(1:length(meshes)),start_poses, IDENTITY_POSE)
GL.view_depth_image(d)

# +
current_poses = copy(start_poses)

#ICP Exploration
get_cloud_func(p,all_poses,idx) = let
    poses = copy(all_poses)
    poses[idx] = p
    d = GL.gl_render(renderer, collect(1:length(meshes)), poses, IDENTITY_POSE)
    c = GL.depth_image_to_point_cloud(d,camera)
    c = T.voxelize(c, voxelize_resolution);
end
i=4
GL.set_intrinsics!(renderer, camera)
new_pose = T.icp_object_pose(current_poses[i], obs_clouds[t+1] , p->get_cloud_func(p,current_poses,i);
    outer_iterations=10, iterations=5);

# new_pose = T.icp_object_pose(new_pose, obs_clouds[t+1] , p->get_cloud_func(p,current_poses,i));

V.reset_visualizer()

V.viz(get_cloud_func(new_pose, current_poses, i) ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(obs_clouds[t+1] ./ 10.0; color=I.colorant"black", channel_name=:h2)
# -

V.reset_visualizer()
V.viz(get_cloud_func(current_poses[i], current_poses, i) ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(obs_clouds[t+1] ./ 10.0; color=I.colorant"black", channel_name=:h2)

# +
c1 = rand(3,50) ./ 5.0
c2 = T.move_points_to_frame_b(c1, Pose(0.0, 1.0, 0.0))

V.reset_visualizer()
V.viz(c1; color=I.colorant"red", channel_name=:h1)
V.viz(c2; color=I.colorant"black", channel_name=:h2)
# -

YCB_DIR = joinpath(dirname(dirname(pwd())),"data")
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR);

# +
c1 = id_to_cloud[5]
c2 = T.move_points_to_frame_b(c1, Pose([0.0, 5.0, 0.0],R.RotXYZ(0.1, 0.1,-0.2)))

V.reset_visualizer()
V.viz(c1; color=I.colorant"red", channel_name=:h1)
V.viz(c2; color=I.colorant"black", channel_name=:h2)
# -

pose = T.icp(c1, c2)
V.reset_visualizer()
V.viz(c1; color=I.colorant"red", channel_name=:h1)
V.viz(T.move_points_to_frame_b(c2,pose); color=I.colorant"black", channel_name=:h2)

c1







# +
current_poses = copy(start_poses)

poses_over_time = []

for t in timesteps
    for i in 1:length(current_poses)
        for _ in 1:2
            
        sweep = -0.3:0.1:0.3
        translated_poses = vcat(IDENTITY_POSE,[
            Pose([x,y,0.0])
            for x in sweep for y in sweep
        ])
        rotated_poses = [Pose(zeros(3),R.RotY(ang)) for ang in -0.5:0.1:0.5]
        current_pose = current_poses[i]
        proposal_translated_poses = (x -> x*current_pose).(translated_poses);
        proposal_rotated_poses = (x -> current_pose*x).(rotated_poses)
        proposal_poses_1 = vcat(proposal_translated_poses, proposal_rotated_poses) 
            
        sweep = -0.1:0.1:0.1
        rotated_translated_poses = vcat((IDENTITY_POSE,IDENTITY_POSE),[
                (Pose([x,y,0.0]), Pose(zeros(3),R.RotX(ang))) 
                
                for x in sweep
                for y in sweep
                for ang in -0.2:0.05:0.2])
            
        current_pose = current_poses[i]
        proposal_poses_2 = (x -> x[1]*current_pose*x[2]).(rotated_translated_poses)
        
            
        proposal_poses = vcat(proposal_poses_1, proposal_poses_2)
        @show length(proposal_poses)
            
        println("render")
        @time images = [
            let
                poses = copy(current_poses)
                poses[i] = p
                d = GL.gl_render(renderer, collect(1:length(meshes)),poses, IDENTITY_POSE)
            end
            for p in proposal_poses
        ];
        println("point cloud")
        @time clouds = map(x->GL.depth_image_to_point_cloud(x,cam), images);
        println("voxelize")
        @time clouds = T.voxelize.(clouds, voxelize_resolution);
        println("score")
        @time scores = (x->score_clouds(obs_clouds[t],x)).(clouds);
        current_poses[i] = proposal_poses[argmax(scores)]; 
        end      
        
    end
    push!(poses_over_time, (t,copy(current_poses)))
end
# -

        println("hi")


# +
i = 10
t, poses = poses_over_time[i]
@show length(poses)
@show length(meshes)
poses = copy(poses)
poses = copy(poses)


d = GL.gl_render(renderer, collect(1:length(meshes)),poses, IDENTITY_POSE)

c = hcat([T.move_points_to_frame_b(x,p) for (x,p) in zip(centered_entities,poses)]...)
V.reset_visualizer()

V.viz(c ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(obs_clouds[t] ./ 10.0; color=I.colorant"black", channel_name=:h2)

GL.view_depth_image(d)

# +
# #ICP Exploration
# get_cloud_func(p,all_poses,idx) = let
#     poses = copy(all_poses)
#     poses[idx] = p
#     d = GL.gl_render(renderer, collect(1:length(meshes)), poses, IDENTITY_POSE)
#     c = GL.depth_image_to_point_cloud(d,camera)
#     c = T.voxelize(c, voxelize_resolution);
# end
# i=1
# GL.set_intrinsics!(renderer, camera)
# new_pose = T.icp_object_pose(p[i], obs_clouds[t] , p->get_cloud_func(p,poses,i));

# V.reset_visualizer()

# V.viz(get_cloud_func(new_pose, poses, i) ./ 10.0; color=I.colorant"red", channel_name=:h1)
# V.viz(obs_clouds[t] ./ 10.0; color=I.colorant"black", channel_name=:h2)
# -


mkdir("video_imgs")

lpad(5,3,"0")

for (i,t) in enumerate(timesteps)
    _, p = poses_over_time[i]
    GL.set_intrinsics!(renderer, camera);
    d = GL.gl_render(renderer, collect(1:length(meshes)),p, IDENTITY_POSE)
    GL.set_intrinsics!(renderer, cam);

    rgb_image = GL.view_rgb_image(rgb_images[t];in_255=true)
    depth_image = GL.view_depth_image(d)
    overlay = T.mix(rgb_image, depth_image, 0.5)

    img = hcat(rgb_image, depth_image, overlay)
    FileIO.save("video_imgs/large_occluder_$(lpad(i-1,4,"0")).png", img)
end



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
goicp = py_goicp.GoICP();
goicp.loadModelAndData(Nm, a_points, Nd, b_points);
goicp.setDTSizeAndFactor(10, 1.0);
goicp.BuildDT();
goicp.Register();
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






