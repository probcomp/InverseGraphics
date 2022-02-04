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

V.setup_visualizer()

YCB_DIR = joinpath(dirname(pwd()),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)

IDX = 500

# Load scene data.
#    gt_poses : Ground truth 6D poses of objects (in the camera frame)
#    ids      : object ids (order corresponds to the gt_poses list)
#    rgb_image, gt_depth_image :
#    cam_pose : 6D pose of camera (in world frame)
#    original_camera : Camera intrinsics
gt_poses, ids, gt_rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
GL.view_rgb_image(gt_rgb_image;in_255=true)

# +
# Overlay
camera = T.scale_down_camera(original_camera, 1)
renderer = GL.setup_renderer(camera, GL.RGBMode())
# Add voxelized object models to renderer instance.
resolution = 0.05
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id])
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
    
#     cloud = id_to_cloud[id]
#     mesh = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer, mesh)
end

# +
colors  = [
    I.colorant"yellow", I.colorant"cyan", I.colorant"lightgreen",
    I.colorant"purple", I.colorant"orange",I.colorant"red", 
]
rgb_image, depth_image = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE; colors=colors)

a = GL.view_rgb_image(rgb_image)
# -

b = GL.view_rgb_image(gt_rgb_image;in_255=true)

T.mix(a,b, 0.7)

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
function get_cloud_from_ids_and_poses(ids, poses, camera_pose)
    depth_image = GL.gl_render(renderer, ids, poses, camera_pose)
    cloud = GL.depth_image_to_point_cloud(depth_image, camera)
    if isnothing(cloud)
        cloud = zeros(3,1)
    end
    cloud
end
V.reset_visualizer()
c = get_cloud_from_ids_and_poses(ids, map(x->T.get_c_relative_to_a(cam_pose,x), gt_poses), cam_pose)
V.viz(T.move_points_to_frame_b(c,cam_pose) ./ 10.0)
# GL.view_depth_image(d)

# Visualize re-rendered depth image of scene.
depth_image = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE)
GL.view_depth_image(clamp.(depth_image, 0.0, 200.0))
# -



# +
# Model parameters
hypers = T.Hyperparams(
    slack_dir_conc=1000.0, # Flush Contact parameter of orientation VMF
    slack_offset_var=0.5, # Variance of zero mean gaussian controlling the distance between planes in flush contact
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
    get_cloud_from_poses_and_idx=
    (poses, idx, p) -> get_cloud_from_ids_and_poses(ids, poses[1:end-1], p), # Get cloud function
    hyperparams=hypers,
    N=1 # Number of meshes to sample for pseudomarginalizing out the shape prior.
);


num_obj = length(params.boxes)
constraints = Gen.choicemap()
constraints[:camera_pose] = cam_pose
constraints[:p_outlier] = params.hyperparams.p_outlier
constraints[:noise] = params.hyperparams.noise

# Intialize the graph to have no edges.
g = T.graph_with_edges(num_obj, [])
constraints[T.structure_addr()] = g

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

V.reset_visualizer()
V.viz(constraints[T.obs_addr()] ./ 10.0)
# -

# Generate a trace
trace, _ = Gen.generate(T.scene, (params,), constraints);
@show Gen.get_score(trace)

# Visulize initial trace.
V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"red", channel_name=:obs_cloud)
# V.viz(T.get_gen_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"black", channel_name=:gen_cloud)

T.get_obs_cloud_in_world_frame(trace)

# +
# Visulize initial trace.
V.reset_visualizer()
gen_cloud = T.get_gen_cloud_in_world_frame(trace) ./ 10.0

V.viz(gen_cloud; color=I.colorant"black", channel_name=:gen_cloud)

for i in 1:size(gen_cloud)[2]
    V.viz_sphere(gen_cloud[:,i],Symbol("sphere_$(i)"); radius=6 * params.hyperparams.resolution * 2 ./ 10.0,color=I.RGBA(1.0, 0.0, 0.0, 0.05))
end

# -



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
        trace, acc = T.icp_move(trace, pose_addr, 3, 1000.0)
        trace, acc = T.icp_move(trace, pose_addr, 3, 1000.0)
    end
end

possible_scene_graphs = T.get_all_possible_scene_graphs(T.get_num_objects(trace); depth_limit=2)
scores = [Gen.get_score(T.force_structure(trace,g)) for g in possible_scene_graphs];
trace, acc = T.full_structure_move(trace, vcat(possible_scene_graphs, 
        [T.get_structure(trace)]), vcat(T.normalize_log_weights(scores)*0.9, [0.1]))
T.get_edges(trace)

for j in 1:T.get_num_objects(trace)-1
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

›››`
# -

possible_scene_graphs = T.get_all_possible_scene_graphs(T.get_num_objects(trace); depth_limit=2)
scores = [Gen.get_score(T.force_structure(trace,g)) for g in possible_scene_graphs];
trace, acc = T.full_structure_move(trace, vcat(possible_scene_graphs, [T.get_structure(trace)]), vcat(T.normalize_log_weights(scores)*0.9, [0.1]))
T.get_edges(trace)

for j in 1:T.get_num_objects(trace)-1
    for _ in 1:100
        trace, acc = Gen.mh(
            trace,
            T.in_place_drift_randomness,
            (j, 0.1, 2000.0, false),
            T.in_place_drift_involution,
            check = false,
        )
    end
    for _ in 1:100
        trace, acc = Gen.mh(
            trace,
            T.in_place_drift_randomness,
            (j, 0.01, 100.0, false),
            T.in_place_drift_involution,
            check = false,
        )
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

# +

@gen function tableWithStacksScene(λ)
  numStacks ~ poisson(λ)
  g = emptyTableScene()
  for i in 1:numStacks
    bottomObj ~ uniform_discrete(possibleBottomObjs)
    bottomX ~ uniform(tabletop_xmin, tabletop_xmax)
    bottomY ~ uniform(tabletop_ymin, tabletop_ymax)
    bottomAngle ~ uniform(0, 2π)
    topObj ~ uniform_discrete(allObjs)
    topX ~ uniform(-3, 3)
    topY ~ uniform(-3, 3)
    topAngle ~ uniform(0, 2π)
    setContact!(g, :tabletop, TOP, bottomObj,
                BOTTOM, bottomX, bottomY, bottomAngle)
    setContact!(g, bottomObj, TOP, topObj,
                BOTTOM, topX, topY, topAngle)
  end
  return g
end


@gen function penalizedTableWithStacksScene(λ)
  g = tableWithStacksScene(λ)
  for (obj1, obj2) in all_pairs(objects(g))
    if interpenetrates(g, obj1, obj2) mean = 100 else mean = 0 end
    interpenetrationPenalty[obj1, obj2] ~ normal(mean, 0.01)
    noisyOcclusionFrac[obj1, obj2] ~ normal(
         occlusionFrac(g, obj1, obj2), 1.0)
  end
end

observations = (
    interpenetrationPenalty = zero_vector,
    noisy_occlusion_frac[:sugarBox, :jello] = 0.5
)




while has_interpenetrating_objs(trace[:sceneGraph])
  (obj1, obj2) = first(interpenetrating_objs(trace[:sceneGraph]))
  Gen.mh!(trace, drift_kernel(:scene => (obj2 => :contactX), scale=5,
                              :scene => (obj2 => :contactY), scale=5))
end

while !occludes(trace[:sceneGraph], :sugarBox1, :jelloBox2)
  Gen.mh!(trace, select(:scene => (:jelloBox2, :contactX),
                        :scene => (:jelloBox2, :contactY)))
  for iter in 1:10
    Gen.mh!(trace, drift_kernel(:scene => (:jelloBox2, :contactX), scale=5,
                                :scene => (:jelloBox2, :contactY), scale=5))
  end
end

for iter in 1:20
  v = gradient(trace, addrs, objective=joint_log_prob)
  trace[addrs] += stepSizes .* unit(v)
end



# -

