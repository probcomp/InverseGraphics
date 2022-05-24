import Revise
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import NearestNeighbors
import GenDirectionalStats as GDS
import Gen
import Open3DVisualizer as V
import MeshCatViz as MV
import Rotations as R
import StaticArrays
import Serialization
import FileIO

# Load YCB objects
YCB_DIR = joinpath(dirname(dirname(pathof(T))),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)

camera_original = T.GL.CameraIntrinsics()
camera = T.scale_down_camera(camera_original, 4)

function setup_renderer()
    # Set up renderer with scaled down intrinsics: 160x120
    renderer = T.GL.setup_renderer(camera, T.GL.DepthMode();gl_version=(4,1))
    resolution = 0.05
    for id in all_ids
        cloud = id_to_cloud[id]
        mesh = T.GL.mesh_from_voxelized_cloud(T.GL.voxelize(cloud, resolution), resolution)
        T.GL.load_object!(renderer, mesh)
    end
    renderer
end

cloud_lookup, rotations_to_enumerate_over, unit_sphere_directions, other_rotation_angle, dirs = Serialization.deserialize("render_caching_data.data")


function get_cloud_cached(p, id, camera)
    cam_pose = T.Pose(zeros(3),R.RotX(-asin(p.pos[2] / p.pos[3])) * R.RotY(asin(p.pos[1] / p.pos[3])))
    adjusted_pose = inv(cam_pose) * p
    idx1 = argmin(sum((dirs .- (adjusted_pose.orientation * [1,0,0])).^2, dims=1))[2]
    idx2 = argmin([abs(R.rotation_angle(inv(adjusted_pose.orientation) * r)) for r in rotations_to_enumerate_over[idx1,:]])
    cached_cloud = T.move_points_to_frame_b(cloud_lookup[id,idx1,idx2], p)
    pixel_coords = T.GL.point_cloud_to_pixel_coordinates(cached_cloud, camera)
    idxs = (0 .< pixel_coords[1,:] .< camera.width) .& (0 .< pixel_coords[2,:] .< camera.height)
    cached_cloud = cached_cloud[:, idxs]
end

MV.setup_visualizer()



renderer = setup_renderer()


# Select random object id.
gt_object_id = rand(all_ids)
# Random pose.
gt_object_pose = T.uniformPose(2.0, 2.0, -2.0, 2.0, 5.0, 20.0);
# Render depth.
gt_depth = T.GL.gl_render(renderer, [gt_object_id], [gt_object_pose], IDENTITY_POSE);
# Convert to point cloud.
gt_cloud = T.GL.depth_image_to_point_cloud(gt_depth, camera)

best_object_id, best_latent_cloud, best_pose, likelihood_scores, _ = T.object_recognition_and_pose_estimation(
    renderer.camera_intrinsics, all_ids,gt_cloud, 0.01, get_cloud_cached);
@show gt_object_pose, best_pose
@show gt_object_id, best_object_id

results_dir = "object_recognition_evaluation"
mkdir(results_dir)

for iter in 1:10000
    # Select random object id.
    gt_object_id = rand(all_ids)
    # Random pose.
    gt_object_pose = T.uniformPose(2.0, 2.0, -2.0, 2.0, 5.0, 20.0);
    # Render depth.
    gt_depth = T.GL.gl_render(renderer, [gt_object_id], [gt_object_pose], IDENTITY_POSE);
    # Convert to point cloud.
    gt_cloud = T.GL.depth_image_to_point_cloud(gt_depth, camera)

    best_object_id, best_latent_cloud, best_pose, likelihood_scores, _ = T.object_recognition_and_pose_estimation(renderer.camera_intrinsics, all_ids,gt_cloud, 0.01, get_cloud_cached);
    Serialization.serialize(joinpath(results_dir, "$(lpad(iter, 6, '0')).data"), (gt_object_id, gt_object_pose, best_object_id, best_pose, likelihood_scores))
end


## Evaluation

right = 0
wrong = 0
wrong_examples = []
for iter in 1:10000
    (gt_object_id, gt_object_pose, best_object_id, best_pose, likelihood_scores) = Serialization.deserialize(joinpath(results_dir, "$(lpad(iter, 6, '0')).data"))
    if gt_object_id == best_object_id
        right += 1
    else
        wrong += 1
        push!(wrong_examples, (gt_object_id, gt_object_pose, best_object_id, best_pose, likelihood_scores))
    end
end
@show right, wrong
@show right / (right + wrong)

counts = zeros(length(all_ids))
for (id,_,_,_,_) in wrong_examples
    counts[id] += 1
end
counts

index = 12
gt_object_id, gt_object_pose, best_object_id, best_pose, likelihood_scores = wrong_examples[index];
# Render depth.
gt_depth = T.GL.gl_render(renderer, [gt_object_id], [gt_object_pose], IDENTITY_POSE);
# Convert to point cloud.
gt_cloud = T.GL.depth_image_to_point_cloud(gt_depth, camera);
@show gt_object_id, best_object_id
@show maximum(likelihood_scores)
MV.reset_visualizer()
MV.viz(gt_cloud ./ 30.0; color=T.I.colorant"black", channel_name=:a)
MV.viz(get_cloud_cached(best_pose, best_object_id) ./ 30.0; color=T.I.colorant"red", channel_name=:b)
@time best_object_id, best_latent_cloud, best_pose, likelihood_scores, _ = T.object_recognition_and_pose_estimation(renderer, all_ids,gt_cloud, 0.01, get_cloud_cached);
@show gt_object_id, best_object_id
@show maximum(likelihood_scores)
MV.reset_visualizer()
MV.viz(gt_cloud ./ 10.0; color=T.I.colorant"black", channel_name=:a)
MV.viz(best_latent_cloud ./ 10.0; color=T.I.colorant"red", channel_name=:b)

clouds = []
for i in 1:100
    gt_object_id, gt_object_pose, best_object_id, best_pose, likelihood_scores = wrong_examples[index];
    # Render depth.
    gt_depth = T.GL.gl_render(renderer, [gt_object_id], [gt_object_pose], IDENTITY_POSE);
    # Convert to point cloud.
    gt_cloud = T.GL.depth_image_to_point_cloud(gt_depth, camera);
    push!(clouds, gt_cloud)
end

V.open_window(camera, T.IDENTITY_POSE)
V.open_window()
for i in 1:100
    V.clear()
    V.make_point_cloud(clouds[i];color=T.I.colorant"red")
    T.FileIO.save("gt_$(i).png", T.GL.view_rgb_image(V.capture_image()))

    gt_object_id, gt_object_pose, best_object_id, best_pose, likelihood_scores = wrong_examples[index];

    V.clear()
    V.make_point_cloud(get_cloud_cached(best_pose, best_object_id);color=T.I.colorant"black")
    T.FileIO.save("pred_$(i).png", T.GL.view_rgb_image(V.capture_image()))
end

index = 11
gt_object_id, gt_object_pose, best_object_id, best_pose, likelihood_scores = wrong_examples[index];
# Render depth.
gt_depth = T.GL.gl_render(renderer, [gt_object_id], [gt_object_pose], IDENTITY_POSE);
# Convert to point cloud.
gt_cloud = T.GL.depth_image_to_point_cloud(gt_depth, camera);



@show Gen.logpdf(
    T.volume_cloud_likelihood,
    T.voxelize(gt_cloud, 0.01),
    best_latent_cloud,
    0.01,
    0.1,
    (-100.0,100.0,-100.0,100.0,-100.0,300.0)
)

