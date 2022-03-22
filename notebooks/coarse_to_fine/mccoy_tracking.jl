# -*- coding: utf-8 -*-
import Revise
import GLRenderer as GL
import Images as I
import ImageView as IV
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

box = GL.box_mesh_from_dims(ones(3));
occluder = GL.box_mesh_from_dims([2.0, 5.0, 0.1]);
intrinsics = GL.CameraIntrinsics();
intrinsics = GL.scale_down_camera(intrinsics, 6)
renderer = GL.setup_renderer(intrinsics, GL.DepthMode())

GL.load_object!(renderer, occluder);
GL.load_object!(renderer, box);

# Example render
occluder_pose = Pose([0.0, 0.0, 10.0], IDENTITY_ORN)
object_pose = Pose([5.0, 0.0, 12.0], IDENTITY_ORN)
i = GL.gl_render(renderer, [1, 2], [occluder_pose, object_pose], IDENTITY_POSE);
x = GL.view_depth_image(i);
IV.imshow(x);
c = GL.depth_image_to_point_cloud(i, intrinsics);
V.viz(c)

resolution = 0.05

# Simulate motion
pose_sequence = [
    Pose([x, 0.0, 12.0], IDENTITY_ORN)
    for x in -5.0:0.25:5.0
];
gt_images = [
    GL.gl_render(renderer, [1, 2], [occluder_pose, object_pose], IDENTITY_POSE)
    for object_pose in pose_sequence
];
x = GL.view_depth_image.(gt_images);
gif = cat(GL.view_depth_image.(gt_images)...; dims=3);
IV.imshow(gif)


Gen.@gen function model(timesteps)
    positions = []
    poses = []
    depth_images = []
    voxelized_clouds = []
    obs_clouds = []
    for i in 1:timesteps
        if i == 1
            x = {(:x, i)} ~ Gen.uniform(-6.0, 6.0)
        else
            x = {(:x, i)} ~ Gen.normal(positions[end], 0.5)
        end
        push!(positions, x)

        object_pose = Pose([x, 0.0, 12.0], IDENTITY_ORN)
        depth_image = GL.gl_render(renderer, [1, 2], [occluder_pose, object_pose], IDENTITY_POSE)
        rendered_cloud = GL.depth_image_to_point_cloud(depth_image, intrinsics)
        voxelized_cloud = GL.voxelize(rendered_cloud, resolution)
        obs_cloud = {(:obs_cloud, i)} ~ T.uniform_mixture_from_template(voxelized_cloud,
            0.01, # p_outlier
            resolution * 2,
            (-100.0, 100.0, -100.0, 100.0, -100.0, 100.0)
        )
        push!(poses, object_pose)
        push!(depth_images, depth_image)
        push!(voxelized_clouds, voxelized_cloud)
        push!(obs_clouds, obs_cloud)
    end
    return (poses=poses, depth_images=depth_images, voxelized_clouds=voxelized_clouds, obs_clouds=obs_clouds)
end
using Gen

function get_obs_cloud(trace, t)
    get_retval(trace).obs_clouds[t]
end

#####
##### Running the model
#####

@time trace, = generate(model, (3,));
clouds = get_retval(trace).voxelized_clouds
V.viz(clouds[1])

gt_clouds = [
    GL.voxelize(GL.depth_image_to_point_cloud(i, intrinsics), resolution)
    for i in gt_images
];
num_timesteps = length(gt_clouds)
observations = choicemap(
    [
        (:obs_cloud, i) => c
        for (i, c) in enumerate(gt_clouds)
    ]...
)
trace, = generate(model, (num_timesteps,), observations);
clouds = get_retval(trace).voxelized_clouds;
V.viz(clouds[7]);
clouds = get_retval(trace).obs_clouds;
V.viz(clouds[1]);

#####
##### Inference with enumerative proposal
#####

mixture_of_normals = HomogeneousMixture(normal, [0, 0])

@gen function initial_proposal(trace, grid_step, gt_clouds)
    x_gridded = -6.0:grid_step:6.0
    weights = [
        let
            choices = choicemap(
                (:obs_cloud, 1) => gt_clouds[t],
                (:x, 1) => x)
            w = assess(model, (1,), choices)
            (w, x)
        end
        for x in x_gridded
    ]
    normalized_weights = exp.(weights .- Gen.logsumexp(weights))
    {(:x, 1)} ~ mixture_of_normals(
        normalized_weights,
        x_gridded,
        0.01 * ones(length(x_gridded))
    )
end

@gen function transition_proposal(trace, t, grid_step, gt_clouds)
    x = trace[(:x, t-1)]
    x_gridded = (x-1.0):grid_step:(x+1.0)
    weights = [
        let
            choices = choicemap(
                (:obs_cloud, 1) => gt_clouds[t],
                (:x, 1) => x
            )
            w = assess(model, (1, ), choices)
            (w, x)
        end
        for x in x_gridded
    ]
    normalized_weights = exp.(weights .- Gen.logsumexp(weights))
    {(:x, t)} ~ mixture_of_normals(
        normalized_weights,
        x_gridded,
        0.01 * ones(length(x_gridded))
    )
end

# Use SMCGF.
using GenSMCGF
smc_gf = SMCGF(model, initial_proposal, transition_proposal)
alg_1_proposal_args = [...]
alg_2_proposal_args = [...]