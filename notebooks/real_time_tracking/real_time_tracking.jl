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

YCB_DIR = joinpath(dirname(dirname(pwd())),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)

SCENE = 49

# Load scene data.
#    gt_poses : Ground truth 6D poses of objects (in the camera frame)
#    ids      : object ids (order corresponds to the gt_poses list)
#    rgb_image, gt_depth_image :
#    cam_pose : 6D pose of camera (in world frame)
#    original_camera : Camera intrinsics
gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, SCENE,1, world_scaling_factor, id_to_shift
);
GL.view_rgb_image(rgb_image;in_255=true)

renderer_original = GL.setup_renderer(original_camera, GL.RGBMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
texture_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id])
    mesh.tex_path = texture_paths[id]
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
    GL.load_object!(renderer_original, mesh)
end

# +
# Create renderer instance
# camera = T.scale_down_camera(original_camera, 4)
camera = T.scale_down_camera(original_camera, 4)
renderer = GL.setup_renderer(camera, GL.DepthMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id])
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
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
rerendered_cloud = get_cloud_from_ids_and_poses(ids, map(x->T.get_c_relative_to_a(cam_pose,x), gt_poses), cam_pose)
depth_image = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE)
x = GL.view_depth_image(clamp.(depth_image, 0.0, 200.0))

# +
# Model parameters
hypers = T.Hyperparams(
    slack_dir_conc=1000.0, # Flush Contact parameter of orientation VMF
    slack_offset_var=0.5, # Variance of zero mean gaussian controlling the distance between planes in flush contact
    p_outlier=0.01, # Outlier probability in point cloud likelihood
    noise=0.06, # Spherical ball size in point cloud likelihood
    resolution=0.03, # Voxelization resolution in point cloud likelihood
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
# dense_poses, dense_ids = T.load_ycbv_dense_fusion_predictions_adjusted(YCB_DIR, SCENE, 1, world_scaling_factor, id_to_shift);
# dense_poses_reordered = [dense_poses[findfirst(dense_ids .== id)]   for id in ids];
GL.activate_renderer(renderer)
for i=1:num_obj-1
   constraints[T.floating_pose_addr(i)] = T.get_c_relative_to_a(cam_pose, gt_poses[i])
end

@time trace, _ = Gen.generate(T.scene, (params,), constraints);
@show Gen.get_score(trace)

V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"black", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"red", channel_name=:gen_cloud)

# -

GL.activate_renderer(renderer_original)
colors = [I.colorant"red",I.colorant"blue",I.colorant"green",I.colorant"yellow"]
GL.view_rgb_image(GL.gl_render(renderer_original, ids, gt_poses, IDENTITY_POSE;colors=colors)[1])

all_images = []
for t in 1:1000
    gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
        YCB_DIR, SCENE,t, world_scaling_factor, id_to_shift
    );
    GL.view_rgb_image(rgb_image;in_255=true)

    GL.activate_renderer(renderer_original)
    i = T.mix(
        GL.view_rgb_image(rgb_image; in_255=true),
        GL.view_rgb_image(GL.gl_render(renderer_original, ids, gt_poses, IDENTITY_POSE;colors=colors)[1]), 0.4)
    GL.activate_renderer(renderer)
    push!(all_images, i)
end



# GL.activate_renderer(renderer_original)
# i = T.mix(
#     GL.view_rgb_image(rgb_image; in_255=true),
#     GL.view_depth_image(GL.gl_render(renderer_original, ids, T.get_poses(trace)[1:end-1], cam_pose)), 0.5)
# GL.activate_renderer(renderer)
# i



# Visualize final trace.
V.reset_visualizer()
V.viz(T.get_obs_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"black", channel_name=:obs_cloud)
V.viz(T.get_gen_cloud_in_world_frame(trace) ./ 10.0; color=I.colorant"red", channel_name=:gen_cloud)

GL.Mesh
