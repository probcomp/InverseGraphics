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


function object_recognition_and_pose_estimation(renderer, gt_cloud, v_resolution, get_cloud_p_id)
    gt_cloud = T.voxelize(gt_cloud, v_resolution)

    # Make KDTree on observed point cloud
    c1_tree = NearestNeighbors.KDTree(gt_cloud);
    centroid = T.centroid(gt_cloud);

    iters = 30
    pose_hypotheses = Matrix{Any}(zeros(length(all_ids), iters))
    latent_clouds = Matrix{Any}(zeros(length(all_ids), iters))
    likelihood_scores = Matrix{Any}(zeros(length(all_ids), iters))

    Threads.@threads for id in all_ids
        Threads.@threads for iter in 1:iters
            # Intial random pose at centroid of observed cloud.
            start_pose = Pose(centroid, GDS.uniform_rot3())
            # Run ICP to refine that initial pose. (Use the KDTree to accelerate this.)
            refined_pose = T.icp_object_pose(
                start_pose,
                gt_cloud,
                p -> get_cloud_p_id(p, id);
                c1_tree=c1_tree,
                outer_iterations=5,
                iterations=2
            );
            # Get (latent) cloud corresponding to object at refined_pose
            c = get_cloud_p_id(refined_pose, id)
            # Compute the probability of the observed cloud being generated from the latent cloud.
            # This is the 3DP3 likelihood.
            score = Gen.logpdf(
                # T.uniform_mixture_from_template,
                T.volume_cloud_likelihood,
                gt_cloud,
                c,
                v_resolution,
                0.1,
                (-100.0,100.0,-100.0,100.0,-100.0,300.0)
            )
            
            pose_hypotheses[id,iter] = refined_pose
            latent_clouds[id,iter] = c
            likelihood_scores[id,iter] = score
        end
    end
    
    # Get best scoring hypothesis, cloud, id
    best_hypothesis_index = argmax(likelihood_scores)
    best_latent_cloud = latent_clouds[best_hypothesis_index]
    best_object_id = best_hypothesis_index[1]

    best_object_id, best_latent_cloud, likelihood_scores[best_hypothesis_index], (;latent_clouds, pose_hypotheses, likelihood_scores)
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

cloud_lookup, rotations_to_enumerate_over, unit_sphere_directions, other_rotation_angle, dirs = Serialization.deserialize("render_caching_data.data")

function get_cloud_cached(p, id)
    cam_pose = T.Pose(zeros(3),R.RotX(-asin(p.pos[2] / p.pos[3])) * R.RotY(asin(p.pos[1] / p.pos[3])))
    idx1 = argmin(sum((dirs .- (p.orientation * [1,0,0])).^2, dims=1))[2]
    idx2 = argmin([abs(R.rotation_angle(inv(p.orientation) * r)) for r in rotations_to_enumerate_over[idx1,:]])
    cached_cloud = T.move_points_to_frame_b(cloud_lookup[id,idx1,idx2], p)
end

MV.setup_visualizer()



renderer = setup_renderer()

# Select random object id.
gt_object_id = rand(all_ids)
# Random pose.
gt_object_pose = T.uniformPose(1.0, 1.0, -1.0, 1.0, 5.0, 12.0);
# Render depth.
gt_depth = T.GL.gl_render(renderer, [gt_object_id], [gt_object_pose], IDENTITY_POSE);
# Convert to point cloud.
gt_cloud = T.GL.depth_image_to_point_cloud(gt_depth, camera)

@time best_object_id, best_latent_cloud, score = object_recognition_and_pose_estimation(renderer, gt_cloud, 0.01, get_cloud_cached);
@show gt_object_id, best_object_id
@show score

MV.reset_visualizer()
MV.viz(gt_cloud ./ 10.0; color=T.I.colorant"black", channel_name=:a)
MV.viz(best_latent_cloud ./ 10.0; color=T.I.colorant"red", channel_name=:b)


V.open_window();
V.add(V.make_point_cloud(gt_cloud ;color=T.I.colorant"red"))
V.add(V.make_point_cloud(best_latent_cloud; color=T.I.colorant"blue"))
V.run()

# renderer = setup_renderer()
# cached_cloud = get_cloud_func_cached(gt_object_pose, gt_object_id);
# non_cached_cloud = get_cloud_func(gt_object_pose, gt_object_id);
# V.open_window();
# V.add(V.make_point_cloud(cached_cloud ;color=T.I.colorant"red"))
# V.add(V.make_point_cloud(non_cached_cloud; color=T.I.colorant"blue"))
# V.run()

