import Revise
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import NearestNeighbors
import GenDirectionalStats as GDS
import Gen
import Open3DVisualizer as V

# Load YCB objects
YCB_DIR = joinpath(dirname(dirname(pathof(T))),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)

function object_recognition_and_pose_estimation(renderer, gt_cloud, v_resolution)
    pose_hypotheses = []
    latent_clouds = []
    likelihood_scores = []
    ids = []

    gt_cloud = T.voxelize(gt_cloud, v_resolution)

    # Make KDTree on observed point cloud
    c1_tree = NearestNeighbors.KDTree(gt_cloud);
    centroid = T.centroid(gt_cloud);


    for id in all_ids
        function get_cloud_func(p)
            c = T.GL.depth_image_to_point_cloud(T.GL.gl_render(renderer, [id], [p], IDENTITY_POSE), camera)
            c = T.voxelize(c, v_resolution)
            c
        end
        
        for _ in 1:20
            # Intial random pose at centroid of observed cloud.
            start_pose = Pose(centroid, GDS.uniform_rot3())
            # Run ICP to refine that initial pose. (Use the KDTree to accelerate this.)
            refined_pose = T.icp_object_pose(start_pose, gt_cloud, get_cloud_func; c1_tree=c1_tree, outer_iterations=3, iterations=5);
            # Get (latent) cloud corresponding to object at refined_pose
            c = get_cloud_func(refined_pose)
            # Compute the probability of the observed cloud being generated from the latent cloud.
            # This is the 3DP3 likelihood.
            score = Gen.logpdf(
                T.uniform_mixture_from_template,
                gt_cloud,
                c,
                resolution,
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

    best_object_id, best_latent_cloud
end


camera = T.scale_down_camera(T.GL.CameraIntrinsics(), 6)
renderer = T.GL.setup_renderer(camera, T.GL.DepthMode();gl_version=(4,1))
resolution = 0.05
for id in all_ids
    cloud = id_to_cloud[id]
    mesh = T.GL.mesh_from_voxelized_cloud(T.GL.voxelize(cloud, resolution), resolution)
    T.GL.load_object!(renderer, mesh)
end

# Select random object id.
gt_object_id = rand(all_ids)
# Random pose.
gt_object_pose = T.uniformPose(1.0, 1.0, -1.0, 1.0, 8.0, 12.0);
# Render depth.
gt_depth = T.GL.gl_render(renderer, [gt_object_id], [gt_object_pose], IDENTITY_POSE);
# Convert to point cloud.
gt_cloud = T.GL.depth_image_to_point_cloud(gt_depth, camera)


@time best_object_id, best_latent_cloud = object_recognition_and_pose_estimation(renderer, gt_cloud, 0.1)
@show gt_object_id, best_object_id

V.open_window();
V.add(V.make_point_cloud(gt_cloud ;color=T.I.colorant"red"))
V.add(V.make_point_cloud(best_latent_cloud; color=T.I.colorant"blue"))
V.run()


