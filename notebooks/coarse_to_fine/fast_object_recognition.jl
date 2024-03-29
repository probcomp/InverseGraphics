import Revise
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import NearestNeighbors
import GenDirectionalStats as GDS
import Gen
import Open3DVisualizer as V
import Rotations as R
import StaticArrays
import Serialization

# Load YCB objects
YCB_DIR = joinpath(dirname(dirname(pathof(T))),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)

function object_recognition_and_pose_estimation(renderer, gt_cloud, v_resolution, get_cloud_p_id)
    pose_hypotheses = []
    latent_clouds = []
    likelihood_scores = []
    ids = []

    gt_cloud = T.voxelize(gt_cloud, v_resolution)

    # Make KDTree on observed point cloud
    c1_tree = NearestNeighbors.KDTree(gt_cloud);
    centroid = T.centroid(gt_cloud);


    Threads.@threads for id in all_ids
        for _ in 1:30
            # Intial random pose at centroid of observed cloud.
            start_pose = Pose(centroid, GDS.uniform_rot3())
            # Run ICP to refine that initial pose. (Use the KDTree to accelerate this.)
            refined_pose = T.icp_object_pose(
                start_pose,
                gt_cloud,
                p -> get_cloud_p_id(p, id);
                c1_tree=c1_tree,
                outer_iterations=4,
                iterations=5
            );
            # Get (latent) cloud corresponding to object at refined_pose
            c = get_cloud_p_id(refined_pose, id)
            # Compute the probability of the observed cloud being generated from the latent cloud.
            # This is the 3DP3 likelihood.
            score = Gen.logpdf(
                T.uniform_mixture_from_template,
                gt_cloud,
                c,
                0.05,
                0.01,
                (-100.0,100.0,-100.0,100.0,-100.0,300.0)
            )
            
            push!(pose_hypotheses, refined_pose)
            push!(latent_clouds, c)
            push!(likelihood_scores, score)
            push!(ids, id)
        end
    end
    
    # Get best scoring hypothesis, cloud, id
    best_hypothesis_index = argmax(likelihood_scores)
    best_latent_cloud = latent_clouds[best_hypothesis_index]
    best_object_id = ids[best_hypothesis_index]

    best_object_id, best_latent_cloud, (;latent_clouds, pose_hypotheses, likelihood_scores, ids)
end

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


renderer = setup_renderer()
# Select random object id.
gt_object_id = rand(all_ids)
# Random pose.
gt_object_pose = T.uniformPose(1.0, 1.0, -1.0, 1.0, 4.0, 6.0);
# Render depth.
gt_depth = T.GL.gl_render(renderer, [gt_object_id], [gt_object_pose], IDENTITY_POSE);
# Convert to point cloud.
gt_cloud = T.GL.depth_image_to_point_cloud(gt_depth, camera)

function get_cloud_func(p, id)
    c = T.GL.depth_image_to_point_cloud(T.GL.gl_render(renderer, [id], [p], IDENTITY_POSE), camera)
    c = T.voxelize(c, 0.05)
    c
end

@time best_object_id, best_latent_cloud, data = object_recognition_and_pose_estimation(renderer, gt_cloud, 0.01, get_cloud_func)
@show gt_object_id, best_object_id

V.open_window(camera_original, IDENTITY_POSE);

import FileIO
V.clear()
V.add(V.make_point_cloud(T.voxelize(gt_cloud, 0.05) ;color=T.I.colorant"red"))
V.set_camera_intrinsics_and_pose(camera_original, IDENTITY_POSE)
V.sync()

img = V.capture_image();
FileIO.save("gt.png", T.GL.view_rgb_image(img))


V.add(V.make_point_cloud(best_latent_cloud; color=T.I.colorant"blue"))
V.set_camera_intrinsics_and_pose(camera_original, IDENTITY_POSE)
V.sync()

img = V.capture_image();
FileIO.save("pred.png", T.GL.view_rgb_image(img))

for i in 1:100
    c = rand(data.latent_clouds)
    V.clear()
    V.add(V.make_point_cloud(T.voxelize(gt_cloud, 0.05) ;color=T.I.colorant"red"))

    V.add(V.make_point_cloud(c; color=T.I.colorant"blue"))
    V.set_camera_intrinsics_and_pose(camera_original, IDENTITY_POSE)
    V.sync()
    img = V.capture_image();
    FileIO.save("/tmp/$(i).png", T.GL.view_rgb_image(img))
end

img_sequence = [FileIO.load("/tmp/$(t).png") for t in 1:100];
gif = cat(img_sequence...;dims=3);
FileIO.save("options.gif", gif);




