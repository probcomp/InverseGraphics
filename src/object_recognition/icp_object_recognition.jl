import GenDirectionalStats: uniform_rot3

function object_recognition_and_pose_estimation(camera, all_ids, gt_cloud, v_resolution, get_cloud_p_id; num_particles=30)
    gt_cloud = voxelize(gt_cloud, v_resolution)

    # Make KDTree on observed point cloud
    c1_tree = NearestNeighbors.KDTree(gt_cloud);
    center = centroid(gt_cloud);

    pose_hypotheses = Matrix{Any}(zeros(length(all_ids), num_particles))
    latent_clouds = Matrix{Any}(zeros(length(all_ids), num_particles))
    likelihood_scores = Matrix{Any}(zeros(length(all_ids), num_particles))

    Threads.@threads for id in all_ids
        Threads.@threads for iter in 1:num_particles
            # Intial random pose at centroid of observed cloud.
            start_pose = Pose(center, uniform_rot3())
            # Run ICP to refine that initial pose. (Use the KDTree to accelerate this.)
            refined_pose = icp_object_pose(
                start_pose,
                gt_cloud,
                p -> get_cloud_p_id(p, id, camera);
                c1_tree=c1_tree,
                outer_iterations=5,
                iterations=3
            );
            # Get (latent) cloud corresponding to object at refined_pose
            c = get_cloud_p_id(refined_pose, id, camera)
            # Compute the probability of the observed cloud being generated from the latent cloud.
            # This is the 3DP3 likelihood.
            score = Gen.logpdf(
                # uniform_mixture_from_template,
                volume_cloud_likelihood,
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
    best_pose = pose_hypotheses[best_hypothesis_index]
    best_object_id = best_hypothesis_index[1]
    @show likelihood_scores[best_hypothesis_index]

    best_object_id, best_latent_cloud, best_pose, likelihood_scores, (;pose_hypotheses, latent_clouds, likelihood_scores)
end