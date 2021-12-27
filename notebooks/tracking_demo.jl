import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
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

gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, 49,1, world_scaling_factor, id_to_shift
);
GL.view_rgb_image(rgb_image;in_255=true)

resolution = 0.5
camera = T.scale_down_camera(original_camera, 1)
renderer = GL.setup_renderer(camera, GL.DepthMode())
for id in all_ids
    cloud = id_to_cloud[id]
    v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer, v, f)
end

camera

size(d)

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
    noise=0.2, resolution=0.5, parent_face_mixture_prob=0.99, floating_position_bounds=(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0))

params = T.SceneModelParameters(
    boxes=vcat([id_to_box[id] for id in ids],[S.Box(40.0,40.0,0.1)]),
    get_cloud_from_poses_and_idx=
    (poses, idx, p) -> get_cloud_from_ids_and_poses(ids, poses[1:end-1], p),
    hyperparams=hypers, N=1)

# +
table_pose = Pose(0.0, 0.0, -0.1)
num_obj = length(params.boxes)
g = T.graph_with_edges(num_obj, [])
constraints = Gen.choicemap((T.structure_addr()) => g, T.floating_pose_addr(num_obj) => table_pose)
for i=1:num_obj-1
   constraints[T.floating_pose_addr(i)] = T.get_c_relative_to_a(cam_pose, gt_poses[i])
end
constraints[:camera_pose] = cam_pose

obs_cloud = GL.depth_image_to_point_cloud(gt_depth_image, original_camera)
obs_cloud = GL.move_points_to_frame_b(obs_cloud, cam_pose)
obs_cloud =obs_cloud[:, 1.2 .< obs_cloud[3, :] .< 40.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[1, :] .< 25.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[2, :] .< 25.0];
constraints[T.obs_addr()] = T.voxelize(obs_cloud, params.hyperparams.resolution)
V.reset_visualizer()
V.viz(constraints[T.obs_addr()])
# -

trace, _ = Gen.generate(T.scene, (params,), constraints);

pose_addr = T.floating_pose_addr(1);
trace, acc = T.drift_move(trace, pose_addr, 0.5, 10.0);

model_args

for i in 1:length(ids)
    if !T.isFloating(T.get_structure(trace),i)
        continue
    end
    pose_addr = T.floating_pose_addr(i)
    for _ in 1:50
        trace, acc = T.drift_move(trace, pose_addr, 0.5, 10.0)
        trace, acc = T.drift_move(trace, pose_addr, 1.0, 100.0)
        trace, acc = T.drift_move(trace, pose_addr, 1.5, 1000.0)
        trace, acc = T.drift_move(trace, pose_addr, 0.1, 1000.0)
        trace, acc = T.drift_move(trace, pose_addr, 0.1, 100.0)
        trace, acc = T.drift_move(trace, pose_addr, 0.5, 5000.0)
        trace, acc = T.pose_flip_move(trace, pose_addr, 1, 1000.0)
        trace, acc = T.pose_flip_move(trace, pose_addr, 2, 1000.0)
        trace, acc = T.pose_flip_move(trace, pose_addr, 3, 1000.0)
    end
end

Gen.get_choices(trace)

V.reset_visualizer()
V.viz(T.get_obs_cloud(trace); color=I.colorant"black", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud(trace); color=I.colorant"red", channel_name=:gen_cloud)


# +
model_args = get_args(trace)
argdiffs = map((_) -> NoChange(), model_args)
fwd_choices = Gen.choicemap()
fwd_choices[T.floating_pose_addr(1)] = IDENTITY_POSE
(new_trace, weight, _, discard) = update(trace,
    model_args, argdiffs, fwd_choices)

# proposal_args_backward = (new_trace, proposal_args...,)
# (bwd_weight, _) = assess(proposal, proposal_args_backward, discard)
# alpha = weight - fwd_weight + bwd_weight
# check && check_observations(get_choices(new_trace), observations)
# -

pose_addr = T.floating_pose_addr(1)
trace, acc = T.drift_move(trace, pose_addr, 0.5, 10.0)



# +

trace, acc = T.drift_move(trace, pose_addr, 0.5, 10.0)

# -



scene_graph = S.SceneGraph()
S.floatingPosesOf(scene_graph)
S.addObject!(scene_graph, :top, id_to_box[1])
