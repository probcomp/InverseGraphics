# -*- coding: utf-8 -*-
import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import NearestNeighbors
import LightGraphs as LG
import Gen
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end
import ImageView as IV

V.setup_visualizer()

YCB_DIR = joinpath(dirname(dirname(pathof(T))),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)

IDX = 200

# Load scene data.
#    gt_poses : Ground truth 6D poses of objects (in the camera frame)
#    ids      : object ids (order corresponds to the gt_poses list)
#    rgb_image, gt_depth_image :
#    cam_pose : 6D pose of camera (in world frame)
#    original_camera : Camera intrinsics
gt_poses, ids, gt_rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
x = GL.view_rgb_image(gt_rgb_image;in_255=true)
IV.imshow(x)

# +
# Create renderer instance
camera = T.scale_down_camera(original_camera, 4)
renderer = GL.setup_renderer(camera, GL.DepthMode())
# Add voxelized object models to renderer instance.
resolution = 0.05
for id in all_ids
    cloud = id_to_cloud[id]
    mesh = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer, mesh)
end
# Helper function to get point cloud from the object ids, object poses, and camera pose
function get_cloud(poses, ids, camera_pose)
    depth_image = GL.gl_render(renderer, ids, poses, camera_pose)
    cloud = GL.depth_image_to_point_cloud(depth_image, camera)
    if isnothing(cloud)
        cloud = zeros(3,1)
    end
    cloud
end
V.reset_visualizer()
c = get_cloud(map(x->T.get_c_relative_to_a(cam_pose,x), gt_poses), ids, cam_pose)
V.viz(T.move_points_to_frame_b(c,cam_pose) ./ 10.0)
# GL.view_depth_image(d)

# Visualize re-rendered depth image of scene.
depth_image = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE)
GL.view_depth_image(clamp.(depth_image, 0.0, 200.0))
# +
cloud = GL.depth_image_to_point_cloud(depth_image, camera)
V.reset_visualizer()
V.viz(cloud ./5.0; color=I.colorant"black", channel_name=:gen_cloud)

cloud2 = GL.depth_image_to_point_cloud(gt_depth_image, original_camera)
V.viz(cloud2 ./5.0; color=I.colorant"red", channel_name=:obs_cloud)
# -

vcat(ids,[1])

# +
# Model parameters
hypers = T.Hyperparams(
    slack_dir_conc=800.0, # Flush Contact parameter of orientation VMF
    slack_offset_var=0.2, # Variance of zero mean gaussian controlling the distance between planes in flush contact
    p_outlier=0.01, # Outlier probability in point cloud likelihood
    noise=0.1, # Spherical ball size in point cloud likelihood
    resolution=0.06, # Voxelization resolution in point cloud likelihood
    parent_face_mixture_prob=0.99, # If the bottom of the can is in contact with the table, the an object
    # in contact with the can is going to be contacting the top of the can with this probability.
    floating_position_bounds=(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0), # When an object is
    # not in contact with another object, its pose is drawn uniformly with position bounded by the given extent.
)

params = T.SceneModelParameters(
    boxes=vcat([id_to_box[id] for id in ids],[S.Box(40.0,40.0,0.1)]), # Bounding box size of objects
    ids=vcat(ids,[-1]),
    get_cloud=
    (poses, ids, cam_pose, i) -> get_cloud(poses, ids, cam_pose), # Get cloud function
    hyperparams=hypers,
    N=1 # Number of meshes to sample for pseudomarginalizing out the shape prior.
);


num_obj = length(params.boxes)
constraints = Gen.choicemap()
constraints[:camera_pose] = cam_pose
constraints[:p_outlier] = params.hyperparams.p_outlier
constraints[:noise] = params.hyperparams.noise

# Intialize the graph to have no edges.
empty_graph = T.graph_with_edges(num_obj, [])
constraints[T.structure_addr()] = empty_graph

# Fix the pose of the table.
table_pose = Pose(0.0, 0.0, -0.1)
constraints[T.floating_pose_addr(num_obj)] = table_pose

# Subtract out the background and table points.
obs_cloud = GL.depth_image_to_point_cloud(gt_depth_image, original_camera)
obs_cloud = GL.move_points_to_frame_b(obs_cloud, cam_pose)
obs_cloud =obs_cloud[:, 0.1 .< obs_cloud[3, :] .< 10.0];
obs_cloud =obs_cloud[:, -2.0 .< obs_cloud[1, :] .< 2.0];
obs_cloud =obs_cloud[:, -25.0 .< obs_cloud[2, :] .< 2.0];
obs_cloud = GL.get_points_in_frame_b(obs_cloud, cam_pose)
obs_cloud = T.voxelize(obs_cloud, params.hyperparams.resolution)
constraints[T.obs_addr()] = obs_cloud

# Initialize object poses to be the outputs of DenseFusion neural pose estimator.
dense_poses, dense_ids = T.load_ycbv_dense_fusion_predictions_adjusted(YCB_DIR, IDX, world_scaling_factor, id_to_shift);
dense_poses_reordered = [dense_poses[findfirst(dense_ids .== id)]   for id in ids];
for i=1:num_obj-1
   constraints[T.floating_pose_addr(i)] = T.get_c_relative_to_a(cam_pose, dense_poses_reordered[i])
end

# Generate a trace
trace, _ = Gen.generate(T.scene, (params,), constraints);
@show Gen.get_score(trace)
# Visulize initial trace.
V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"red", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"black", channel_name=:gen_cloud)
# -

# Generate a trace
trace, _ = Gen.generate(T.scene, (params,), constraints);
@show Gen.get_score(trace)
# Visulize initial trace.
V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"red", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"black", channel_name=:gen_cloud)

trace = T.force_structure(trace, empty_graph);

# +
for i in 1:num_obj-1
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

    trace, acc = T.icp_move(trace, i; iterations=10)
end

# Full graph involutive move. Note this only proposes fully connected graphs.
possible_scene_graphs = T.get_all_possible_scene_graphs(T.get_num_objects(trace); depth_limit=2)
scores = [Gen.get_score(T.force_structure(trace,g)) for g in possible_scene_graphs];
trace, acc = T.full_structure_move(trace, vcat(possible_scene_graphs, 
        [T.get_structure(trace)]), vcat(T.normalize_log_weights(scores)*0.9, [0.1]))
@show T.get_edges(trace)

# Involutive moves that propose to turn single edges on and off.
# We first make proposals to consider edges between objects and the table. (the last object is always the table).
for i in 1:num_obj-1
    trace, acc = T.toggle_edge_move(trace, i, num_obj);
end
for i in 1:num_obj-1
    trace, acc = T.toggle_edge_move(trace, i);
end
@show T.get_edges(trace)

for j in 1:T.get_num_objects(trace)-1
    if T.isFloating(T.get_structure(trace),j)
        continue
    end

    for _ in 1:100
        trace, acc = T.in_place_drift_involution_move(trace, j, 0.1, 2000.0)
    end
    for _ in 1:100
        trace, acc = T.in_place_drift_involution_move(trace, j, 0.01, 100.0)
    end
    address(symbol) = T.contact_addr(j, symbol)

    for _ in 1:50
        trace, _ = T.drift_move(trace, address(:x), 0.4)
        trace, _ = T.drift_move(trace, address(:y), 0.4)
        trace, _ = T.drift_move(trace, address(:slack_offset), 0.4)
        trace, _ = Gen.mh(trace, Gen.select(address(:slack_dir), address(:slack_offset)))
        trace, _ = Gen.mh(trace, Gen.select(address(:angle)))
    end
end


# Visualize final trace.
V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"black", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"red", channel_name=:gen_cloud)


