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
import GenDirectionalStats as GDS
import Gen
using UnicodePlots
using StatsBase
using Rotations
using Gen
using GenInferenceDiagnostics
using GenSMCGF
using GenAIDE
using FileIO
import MeshCatViz as V

#V.setup_visualizer()

box = GL.box_mesh_from_dims(ones(3));
occluder = GL.box_mesh_from_dims([2.0, 5.0, 0.1]);
intrinsics = GL.CameraIntrinsics();
intrinsics = GL.scale_down_camera(intrinsics, 7)
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

resolution = 0.6

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

@gen function kernel(t, null, 
        positions, poses, 
        depth_images, voxelized_clouds, obs_clouds)
    if t == 1
        x ~ Gen.uniform(-6.0, 6.0)
    else
        x ~ Gen.normal(positions[end], 0.5)
    end
    push!(positions, x)
    object_pose = Pose([x, 0.0, 12.0], IDENTITY_ORN)
    depth_image = GL.gl_render(renderer, [1, 2], 
                               [occluder_pose, object_pose], IDENTITY_POSE)
    rendered_cloud = GL.depth_image_to_point_cloud(depth_image, intrinsics)
    voxelized_cloud = GL.voxelize(rendered_cloud, resolution)
    obs_cloud ~ T.uniform_mixture_from_template(voxelized_cloud,
                                                0.01, # p_outlier
                                                resolution * 3,
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
    m ~ unf(t, 0, positions,
            poses, depth_images, 
            voxelized_clouds, obs_clouds)
    return (; poses, depth_images, voxelized_clouds, obs_clouds)
end

Gen.@load_generated_functions()

#####
##### Running the model
#####

range = collect(-5.0 : 0.25 : -3.0)
l = length(range)
chm1 = choicemap(((:m => i => :x, x) for (i, x) in enumerate(range))...)
range1 = collect(1.0 : 0.25 : 5.0)
chm2 = choicemap(((:m => i + l => :x, x) for (i, x) in enumerate(range1))...)
chm = merge(chm1, chm2)
@time trace, = generate(model, (45,), chm);
depth_images = get_retval(trace).depth_images
x = GL.view_depth_image.(depth_images);
gif = cat(GL.view_depth_image.(depth_images)...; dims=3);
FileIO.save("anims/1d_occluder_$(gensym()).gif", gif)

#####
##### Inference with enumerative proposal
#####

mixture_of_normals = HomogeneousMixture(normal, [0, 0])

# An enumerative proposal over linear translations.
@gen function initial_proposal(grid_step, chm)
    x_gridded = -6.0 : grid_step : 6.0
    weights = map(x_gridded) do x
        choices = choicemap((:m => 1 => :obs_cloud, chm[:m => 1 => :obs_cloud]),
                            (:m => 1 => :x, x))
        tr, w = generate(model, (1, ), choices)
        w
    end
    normalized_weights = exp.(weights .- Gen.logsumexp(weights))
    {:m => 1 => :x} ~ mixture_of_normals(normalized_weights,
                                         x_gridded,
                                         0.01 * ones(length(x_gridded)))
end

# An enumerative proposal over linear translations in local grid.
@gen function transition_proposal(trace, t, grid_step, chm)
    prev_chm = get_choices(trace)
    prev = get_choices(trace)[:m => t - 1 => :x]
    x_gridded = (prev - 1.0) : grid_step : (prev + 1.0)
    weights = map(x_gridded) do x
        choices = choicemap((:m => t => :obs_cloud, chm[:m => t => :obs_cloud]),
                            (:m => t => :x, x))
        c = merge(prev_chm, choices)
        tr, w  = generate(model, (t, ), c)
        w
    end
    normalized_weights = exp.(weights .- Gen.logsumexp(weights))
    {:m => t => :x} ~ mixture_of_normals(normalized_weights,
                                         x_gridded,
                                         0.01 * ones(length(x_gridded)))
end

@gen function rejuv_proposal(tr, t::Int)
    x = tr[:m => t => :x]
    {:m => t => :x} ~ normal(x, 0.2)
end

@kern function my_kernel(tr, t::Int)
    tr ~ mh(tr, rejuv_proposal, (t, ))
end
    
smc_gf = GenSMCGF.SMCGF(model, 
                        initial_proposal, transition_proposal, 
                        my_kernel)

#####
##### SBC
#####

import Gen: _fill_array!
function _fill_array!(r::QuatRotation{Float64},
        v::Vector{Float64},
        d::Int64)
    svector = Rotations.params(r)
    return _fill_array!(Vector(svector), v, d)
end

function run_msbc(N, N_inf, N_particles, N_rejuvenation, num_timesteps, grid_step)
    population_msbc = [ simulate(model, (num_timesteps,)) for _ in 1 : N]
    probe_msbc = Gen.select((:m => i => :x for i in 1 : num_timesteps)...)
    obs_msbc = Gen.select((:m => i => :obs_cloud for i in 1 : num_timesteps)...)
    function inference_msbc(obs)
        smc_gf = GenSMCGF.SMCGF(model, 
                                initial_proposal, transition_proposal, 
                                my_kernel)
        chms = map(1 : num_timesteps) do t
            addr = :m => t => :obs_cloud
            choicemap((addr, obs[addr]))
        end
        proposal_args = Tuple[(t, grid_step, obs) for t in 2 : num_timesteps]
        pushfirst!(proposal_args, (grid_step, obs))
        args = [(t, ) for t in 1 : num_timesteps]
        argdiffs = [(Gen.IntDiff(1), ) for t in 1 : num_timesteps]
        @time smc_tr = simulate(smc_gf, (chms, args, argdiffs, 
                                         proposal_args, args, 
                                         N_particles, N_rejuvenation))
        particles = smc_tr.population
        log_weights = smc_tr.log_weights
        log_total_weight = Gen.logsumexp(log_weights)
        normalized_weights = exp.(log_weights .- log_total_weight)
        rets = [particles[categorical(normalized_weights)] for _ in 1 : length(particles)]
        return rets
    end
    summary = GenInferenceDiagnostics.msbc(population_msbc,
                                           probe_msbc,
                                           obs_msbc,
                                           inference_msbc)
    return summary
end

function run_sbc(N, N_inf, N_particles, num_timesteps)
    population_sbc = [ simulate(model, (num_timesteps,)) for _ in 1 : N]
    probe_sbc = Gen.select(:m => 1 => :x)
    obs_sbc = Gen.select((:m => i => :obs_cloud 
                          for i in 1 : num_timesteps)...)

    function inference_sbc(obs)
        ret = []
        for k in 1 : N_inf
            trs, lnw = importance_sampling(model, (num_timesteps, ), obs, N_particles)
            nw = exp.(lnw)
            @info StatsBase.mean(nw)
            push!(ret, trs[categorical(nw)])
        end
        return ret
    end

    summary = GenInferenceDiagnostics.sbc(population_sbc,
                                          probe_sbc,
                                          obs_sbc,
                                          inference_sbc)
    return summary
end
