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

# Set up renderer with scaled down intrinsics: 160x120
camera = T.scale_down_camera(T.GL.CameraIntrinsics(), 4)
renderer = T.GL.setup_renderer(camera, T.GL.DepthMode())
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

# Make KDTree on observed point cloud
c1_tree = NearestNeighbors.KDTree(gt_cloud);

centroid = T.centroid(gt_cloud);

pose_hypothesis = []
latent_cloud = []
likelihood_score = []
ids = []
for id in all_ids
    get_cloud_func(p) = T.GL.depth_image_to_point_cloud(T.GL.gl_render(renderer, [id], [p], IDENTITY_POSE), camera)
    
    for _ in 1:20
        # Intial random pose at centroid of observed cloud.
        start_pose = Pose(centroid, GDS.uniform_rot3())
        # Run ICP to refine that initial pose. (Use the KDTree to accelerate this.)
        refined_pose = T.icp_object_pose(start_pose, gt_cloud, get_cloud_func; c1_tree=c1_tree);
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
        
        push!(pose_hypothesis, refined_pose)
        push!(latent_cloud, c)
        push!(likelihood_score, score)
        push!(ids, id)
    end
end

# Get best scoring hypothesis
best_hypothesis_index = argmax(likelihood_score)
best_object_id = ids[best_hypothesis_index]
@show gt_object_id, best_object_id

# Visualize best latent cloud
best_latent_cloud = latent_cloud[best_hypothesis_index]
V.open_window();
V.add(V.make_point_cloud(gt_cloud ;color=T.I.colorant"red"))
V.add(V.make_point_cloud(best_latent_cloud; color=T.I.colorant"blue"))
V.run()


