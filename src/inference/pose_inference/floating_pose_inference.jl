@gen function propose_to_pose(trace, pose_addr, pose, pos_var, rot_conc)
    {pose_addr} ~ gaussianVMF(pose, pos_var, rot_conc)
end
pose_proposal_move(trace, pose_addr, pose, pose_var, rot_conc) = mh(trace, propose_to_pose, (pose_addr, pose, pose_var, rot_conc))

@gen function gaussian_drift_proposal(trace, pose_addr, pos_var, rot_conc)
    {pose_addr} ~ gaussianVMF(trace[pose_addr], pos_var, rot_conc)
end
drift_move(trace, pose_addr, pos_var, rot_conc) = mh(trace, gaussian_drift_proposal, (pose_addr, pos_var, rot_conc))

@gen function gaussian_drift_scalar(trace, pose_addr, var)
    {pose_addr} ~ normal(trace[pose_addr], var)
end
drift_move(trace, pose_addr, var) = mh(trace, gaussian_drift_scalar, (pose_addr, var))


export propose_to_pose, gaussian_drift_proposal, pose_proposal_move, drift_move

# Proposal to rotate an object's pose by 180 degrees about a specified axis.
@gen function pose_flip_proposal(trace, pose_addr, dimension::Int, rot_conc)
    pose = trace[pose_addr]
    euler = zeros(3)
    euler[dimension] = pi
    flipped_pose = get_c_relative_to_a(pose, Pose(R.RotXYZ(euler...)))
    {pose_addr} ~ gaussianVMF(flipped_pose, 0.0001, rot_conc)
end
pose_flip_move(trace, pose_addr, dimension, rot_conc) = mh(trace, pose_flip_proposal, (pose_addr, dimension, rot_conc))

export pose_flip_move



# Pose involution proposal.
@gen function pose_inv_randomness(trace, pose_addr, modifier)
    bit ~ bernoulli(0.5)
    bit
end
function pose_involution(
    trace,
    fwd_choices,
    fwd_ret,
    proposal_args;
    check = false,
)
    pose_addr, modifier = proposal_args
    bit = fwd_ret
    bwd_choices = choicemap(:bit => !bit)

    if bit
        new_pose = get_c_relative_to_a(trace[pose_addr], modifier)
    else
        new_pose = get_c_relative_to_a(trace[pose_addr], inverse_pose(modifier))
    end

    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    new_trace, weight, _, _ = Gen.update(trace, args, argdiffs, choicemap(pose_addr => new_pose))
    return (new_trace, bwd_choices, weight)
end
function pose_involution_move(trace, pose_addr, modifier; iterations=1)
    acceptances = false
    for _ in 1:iterations
        trace, acc = mh(
            trace,
            pose_inv_randomness,
            (pose_addr, modifier,),
            pose_involution,
            check = false,
        )
        acceptances = acc || acceptances
    end
    trace, acceptances
end

export pose_involution_move


# Mixture of Gaussian VMF proposal

@gen function propose_pose_mixture(trace, addr, poses, weights, posVar, rotConc)
    {addr} ~ gaussianVMF_weighted_mixture(poses, weights, posVar, rotConc)
end
pose_mixture_move(trace, addr, poses, weights, posVar, rotConc) = mh(trace, propose_pose_mixture, (addr, poses, weights,  posVar, rotConc))

export pose_mixture_move


# ICP move
function icp_move(trace, i, get_cloud_func; iterations=10, unexplained_radius=0.2, crazy_pose=Pose([-100.0,-100.0,-100.0]))
    # get_cloud_func needs to give the points in the world frame
    model_params = get_args(trace)
    cam_pose = trace[:camera_pose]

    addr = floating_pose_addr(i)
    t, = Gen.update(trace, Gen.choicemap(
            floating_pose_addr(i) => crazy_pose)
    )
    obs_cloud = move_points_to_frame_b(
        get_unexplained_obs_cloud(t, unexplained_radius)[1], cam_pose)

    refined_pose = trace[addr]
    refined_pose = icp_object_pose(
        refined_pose,
        obs_cloud,
        get_cloud_func;
    )

    acceptances = false
    for _ in 1:iterations
        trace, acc = pose_mixture_move(
            trace, addr, [trace[addr], refined_pose], [0.5, 0.5], 0.001, 5000.0
        )
        acceptances = acc || acceptances
    end
    trace, acceptances, refined_pose
end