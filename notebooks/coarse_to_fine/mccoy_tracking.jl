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
using Gen
using GenInferenceDiagnostics
using GenSMCGF
using GenAIDE
import MeshCatViz as V

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
#IV.imshow(x);
c = GL.depth_image_to_point_cloud(i, intrinsics);
#V.viz(c)

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
#IV.imshow(gif)

#####
##### Model
#####

@gen (static) function branch1(positions)
    x ~ Gen.uniform(-6.0, 6.0)
    return x
end

@gen (static) function branch2(positions)
    x ~ Gen.normal(positions[end], 0.5)
    return x
end

sw = Gen.Switch(branch2, branch1)

@gen function kernel(t, null, 
        positions, poses, depth_images, voxelized_clouds, obs_clouds)
    q = Int(t == 1) + 1
    switch ~ sw(q, positions)
    x = switch
    push!(positions, x)
    object_pose = Pose([x, 0.0, 12.0], IDENTITY_ORN)
    depth_image = GL.gl_render(renderer, [1, 2], [occluder_pose, object_pose], IDENTITY_POSE)
    rendered_cloud = GL.depth_image_to_point_cloud(depth_image, intrinsics)
    voxelized_cloud = GL.voxelize(rendered_cloud, resolution)
    obs_cloud ~ T.uniform_mixture_from_template(voxelized_cloud,
                                                0.01, # p_outlier
                                                resolution * 2,
                                                (-100.0, 100.0, -100.0, 100.0, -100.0, 100.0))
    push!(poses, object_pose)
    push!(depth_images, depth_image)
    push!(voxelized_clouds, voxelized_cloud)
    push!(obs_clouds, obs_cloud)
    return nothing
end

unf = Unfold(kernel)

@gen (static) function model(t)
    positions = []
    poses = []
    depth_images = []
    voxelized_clouds = []
    obs_clouds = []
    m ~ unf(t, 0, positions, poses, depth_images, voxelized_clouds, obs_clouds)
    return (; poses, depth_images, voxelized_clouds, obs_clouds)
end

Gen.@load_generated_functions()

#####
##### Running the model
#####

@time trace, = generate(model, (3,));

#####
##### Inference with enumerative proposal
#####

mixture_of_normals = HomogeneousMixture(normal, [0, 0])

@gen function initial_proposal(grid_step, gt_clouds)
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

#####
##### SBC
#####

function run_msbc(N, N_inf, N_particles, num_timesteps)
    population_msbc = [ simulate(model, (num_timesteps,)) for _ in 1 : N]
    probe_msbc = Gen.select((:m => i => :switch => :x for i in 1 : num_timesteps)...)
    obs_msbc = Gen.select((:m => i => :obs_cloud for i in 1 : num_timesteps)...)
    function inference_msbc(obs)
        smc_gf = GenSMCGF.SMCGF(model)
        chms = map(1 : num_timesteps) do t
            addr = :m => t => :obs_cloud
            choicemap((addr, obs[addr]))
        end
        args = [(t, ) for t in 1 : num_timesteps]
        argdiffs = [(Gen.IntDiff(1), ) for t in 1 : num_timesteps]
        trs = [simulate(smc_gf, (chms, args, argdiffs, N_particles)) for _ in 1 : N_inf]
        return trs
    end
    @time summary = GenInferenceDiagnostics.msbc(population_msbc,
                                                 probe_msbc,
                                                 obs_msbc,
                                                 inference_msbc)
    return summary
end
