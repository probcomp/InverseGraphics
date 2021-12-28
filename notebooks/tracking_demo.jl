import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import NearestNeighbors
import Gen
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end

V.setup_visualizer()

YCB_DIR = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2"
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));

SCENE = 50

gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, SCENE,1, world_scaling_factor, id_to_shift
);
GL.view_rgb_image(rgb_image;in_255=true)

# +
resolution = 0.5
camera = T.scale_down_camera(original_camera, 4)
renderer = GL.setup_renderer(camera, GL.DepthMode())
for id in all_ids
    cloud = id_to_cloud[id]
    v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer, v, f)
end

function get_cloud_from_ids_and_poses(ids, poses, camera_pose)
    depth_image = GL.gl_render(renderer, ids, poses, camera_pose)
    cloud = GL.depth_image_to_point_cloud(depth_image, camera)
    if isnothing(cloud)
        cloud = zeros(3,1)
    end
    cloud
end
c = get_cloud_from_ids_and_poses(ids, map(x->T.get_c_relative_to_a(cam_pose,x), gt_poses), cam_pose)
V.viz(T.move_points_to_frame_b(c,cam_pose))
# GL.view_depth_image(d)

# +
hypers = T.Hyperparams(slack_dir_conc=300.0, slack_offset_var=0.5, p_outlier=0.01, 
    noise=0.1, resolution=0.5, parent_face_mixture_prob=0.99, floating_position_bounds=(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0))

params = T.SceneModelParameters(
    boxes=vcat([id_to_box[id] for id in ids],[S.Box(40.0,40.0,0.1)]),
    get_cloud_from_poses_and_idx=
    (poses, idx, p) -> get_cloud_from_ids_and_poses(ids, poses[1:end-1], p),
    hyperparams=hypers, N=1
);

table_pose = Pose(0.0, 0.0, -0.1)
num_obj = length(params.boxes)
g = T.graph_with_edges(num_obj, [])
constraints = Gen.choicemap((T.structure_addr()) => g, T.floating_pose_addr(num_obj) => table_pose)
for i=1:num_obj-1
   constraints[T.floating_pose_addr(i)] = T.get_c_relative_to_a(cam_pose, gt_poses[i])
end
constraints[:camera_pose] = cam_pose
constraints[:p_outlier] = params.hyperparams.p_outlier
constraints[:noise] = params.hyperparams.noise
constraints[:camera_pose] = cam_pose

obs_cloud = GL.depth_image_to_point_cloud(gt_depth_image, original_camera)
obs_cloud = GL.move_points_to_frame_b(obs_cloud, cam_pose)
obs_cloud =obs_cloud[:, 1.2 .< obs_cloud[3, :] .< 40.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[1, :] .< 25.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[2, :] .< 25.0];
obs_cloud = GL.get_points_in_frame_b(obs_cloud, cam_pose)
obs_cloud = T.voxelize(obs_cloud, params.hyperparams.resolution)
constraints[T.obs_addr()] = obs_cloud
V.reset_visualizer()
V.viz(constraints[T.obs_addr()])
# -

trace, _ = Gen.generate(T.scene, (params,), constraints);
Gen.get_score(trace)

V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(trace) ./ 2.0; color=I.colorant"black", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud_in_world_frame(trace) ./ 2.0; color=I.colorant"red", channel_name=:gen_cloud)


# +
inferred_poses = []
trace_history = []
timesteps = []
for time in 2:5:1800
    gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
        YCB_DIR, SCENE, time, world_scaling_factor, id_to_shift
    );
    GL.view_rgb_image(rgb_image;in_255=true)

    obs_cloud = GL.depth_image_to_point_cloud(gt_depth_image, original_camera)
    obs_cloud = GL.move_points_to_frame_b(obs_cloud, cam_pose)
    obs_cloud =obs_cloud[:, 1.2 .< obs_cloud[3, :] .< 40.0];
    obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[1, :] .< 25.0];
    obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[2, :] .< 25.0];
    obs_cloud = GL.get_points_in_frame_b(obs_cloud, cam_pose)
    obs_cloud = T.voxelize(obs_cloud, params.hyperparams.resolution)
    constraints = Gen.choicemap((T.obs_addr(), obs_cloud))
    for i=1:num_obj-1
       constraints[T.floating_pose_addr(i)] = T.get_c_relative_to_a(cam_pose, gt_poses[i])
    end

    trace, = Gen.update(trace, constraints);
    @show Gen.get_score(trace)
    
    function get_cloud_func(p)
        T.voxelize(
            get_cloud_from_ids_and_poses(ids, T.get_poses(trace)[1:end-1], p),
            params.hyperparams.resolution
        )
    end

    
    for _ in 1:10
        new_camera_pose = T.icp_camera_pose(trace[T.camera_pose_addr()], obs_cloud, get_cloud_func)
        addr = T.camera_pose_addr()
        trace, = Gen.update(trace, Gen.choicemap((T.camera_pose_addr(), new_camera_pose)))
#         for _ in 1:5
#             trace, acc = T.pose_mixture_move(trace, addr, [trace[addr], new_camera_pose], [0.5, 0.5], 0.01, 5000.0)
#         end
    end

#     pose_addr = T.camera_pose_addr()
#     for _ in 1:100
#         trace, acc = T.drift_move(trace, pose_addr, 1.0, 100.0)
#         trace, acc = T.drift_move(trace, pose_addr, 1.5, 1000.0)
#         trace, acc = T.drift_move(trace, pose_addr, 0.5, 5000.0)
#     end
    
    V.reset_visualizer()
    V.viz(T.get_obs_cloud_in_world_frame(trace); color=I.colorant"black", channel_name=:obs_cloud)
    V.viz(T.get_gen_cloud_in_world_frame(trace); color=I.colorant"red", channel_name=:gen_cloud)

    push!(inferred_poses, trace[T.camera_pose_addr()])
    push!(trace_history, trace)
    push!(timesteps, time)
end
# -

import Serialization
Serialization.serialize("data.save2", (inferred_poses, trace_history, timesteps))
# inferred_poses, trace_history, timesteps= Serialization.deserialize("data.save");

# +
idx = 610
trace = trace_history[idx]
time = timesteps[idx]

gt_poses, ids, gt_rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, SCENE, time, world_scaling_factor, id_to_shift
);

V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"black", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"red", channel_name=:gen_cloud)

    function get_cloud_func(p)
        T.voxelize(
            get_cloud_from_ids_and_poses(ids, T.get_poses(trace)[1:end-1], p),
            params.hyperparams.resolution
        )
    end
# -

@show Gen.get_score(trace)
tr, = Gen.update(trace, Gen.choicemap((T.camera_pose_addr(), cam_pose)))
@show Gen.get_score(tr)
V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(tr) ./ 10.0; color=I.colorant"black", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud_in_world_frame(tr) ./ 10.0; color=I.colorant"red", channel_name=:gen_cloud)

# +
new_camera_pose = T.icp_camera_pose(
    trace[T.camera_pose_addr()],
    trace[T.obs_addr()], get_cloud_func)
trace, = Gen.update(trace, Gen.choicemap((T.camera_pose_addr(), new_camera_pose)))

V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"black", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"red", channel_name=:gen_cloud)
# -

pose_addr = T.camera_pose_addr()
for _ in 1:100
#     trace, acc = T.drift_move(trace, pose_addr, 1.0, 100.0)
#     trace, acc = T.drift_move(trace, pose_addr, 1.5, 1000.0)
    trace, acc = T.drift_move(trace, pose_addr, 0.5, 1000.0)
end

V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"black", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"red", channel_name=:gen_cloud)

c1 = T.get_obs_cloud_in_world_frame(trace)
c2 = T.get_gen_cloud_in_world_frame(trace)
trans = T.icp(c1,c2)
V.viz(c1 ./ 10.0; color=I.colorant"black", channel_name=:obs_cloud)
V.viz(T.move_points_to_frame_b(c2,trans) ./ 10.0; color=I.colorant"red", channel_name=:gen_cloud)
# V.viz(c2 ./ 10.0; color=I.colorant"red", channel_name=:gen_cloud)



# +
renderer = GL.setup_renderer(original_camera, GL.RGBBasicMode())
# for id in all_ids
#     cloud = id_to_cloud[id]
#     v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
#     GL.load_object!(renderer, v, n, f)
# end
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)

for id in all_ids
    v,n,f,t = renderer.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    
    GL.load_object!(renderer, v, n, f
    )
end

# +
idx = 1
tr = trace_history[idx]
time = timesteps[idx]
gt_poses, ids, gt_rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, SCENE, time, world_scaling_factor, id_to_shift
);


poses = T.get_poses(tr)
colors  = [
    I.colorant"yellow", I.colorant"cyan", I.colorant"lightgreen",
    I.colorant"red", I.colorant"purple", I.colorant"orange"
]
rgb_image,_ = GL.gl_render(renderer, ids, poses, colors, tr[T.camera_pose_addr()])
img = GL.view_rgb_image(gt_rgb_image;in_255=true)
overlay = GL.view_rgb_image(rgb_image)
alpha = 0.7
alpha * overlay .+ (1 - alpha ) * img
full_img = alpha * overlay .+ (1 - alpha ) * img
FileIO.save("tracking_video/$(lpad(idx,4,"0")).png",full_img)

# -

import FileIO
for idx in 1:length(trace_history)
    tr = trace_history[idx]
    time = timesteps[idx]
    gt_poses, ids, gt_rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
        YCB_DIR, SCENE, time, world_scaling_factor, id_to_shift
    );


    poses = T.get_poses(tr)
    colors  = [
        I.colorant"yellow", I.colorant"cyan", I.colorant"lightgreen",
        I.colorant"red", I.colorant"purple", I.colorant"orange"
    ]
    rgb_image,_ = GL.gl_render(renderer, ids, poses, colors, tr[T.camera_pose_addr()])
    img = GL.view_rgb_image(gt_rgb_image;in_255=true)
    overlay = GL.view_rgb_image(rgb_image)
    alpha = 0.7
    full_img = alpha * overlay .+ (1 - alpha ) * img
    FileIO.save("tracking_video/$(lpad(idx,4,"0")).png",full_img)
end




