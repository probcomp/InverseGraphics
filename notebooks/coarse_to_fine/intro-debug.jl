# -*- coding: utf-8 -*-
import Pkg
Pkg.activate("../../");
Pkg.status()

import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import NearestNeighbors
import LightGraphs as LG
import StaticArrays
import ProgressMeter
using Distributions
import Gen
using Plots
using ProgressMeter
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end

# +
# # Initialize the renderer
# V.setup_visualizer()
# -

# Loading the YCB object models
YCB_DIR = joinpath(dirname(dirname(pwd())),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)

# +
# Initialize the canera intrinsics and renderer that will render using those intrinsics.
camera = GL.CameraIntrinsics()
# renderer = GL.setup_renderer(camera, GL.DepthMode())
# renderer = GL.setup_renderer(camera, GL.DepthMode();gl_version=(4,1))
resolution = 0.05
# for id in all_ids
#     cloud = id_to_cloud[id]
#     mesh = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution);
#     GL.load_object!(renderer, mesh)
# end


box_mesh = GL.box_mesh_from_dims([2.0, 2.0, 2.0])
GL.load_object!(renderer, box_mesh)

@show camera;
# -



# ## introductory visualizations
# to get a better sense of resolution effect on likelihood + qualitative results 

# +
# #########
# # radius scaling
# #########
# bounds = (-100.0, 100.0, -100.0, 100.0,-100.0,100.0)

# i = 1
# pose = Pose([0.0, 0.0, 3.0], R.RotXYZ(0.4, -0.2, 0.4))

# # Reset the intrinsics inside of the renderer.
# GL.set_intrinsics!(renderer, camera) 

# # And render the same image as above.
# gt_depth_image = GL.gl_render(renderer, [i], [pose], IDENTITY_POSE)
# IJulia.display(GL.view_depth_image(gt_depth_image))

# # Create point cloud corresponding to that rendered depth image.
# gt_cloud = GL.depth_image_to_point_cloud(gt_depth_image, camera)
# # print(size(gt_cloud))


# @show i

# radius = 0.01  # "tight" radius
# sampled_cloud = Gen.random(T.uniform_mixture_from_template, gt_cloud, 0.0001, radius, bounds)
# # Visualize that point cloud.
# V.setup_visualizer()
# V.viz(sampled_cloud)  # = V.viz(T.move_points_to_frame_b(c, camera_pose))

 
# radius = float(1)  # "loose" radius (can still pick out general shape)
# sampled_cloud = Gen.random(T.uniform_mixture_from_template, gt_cloud, 0.0001, radius, bounds)
# # Visualize that point cloud.
# V.setup_visualizer()
# V.viz(sampled_cloud)  # = V.viz(T.move_points_to_frame_b(c, camera_pose))

 
# radius = float(10)  # very loose radius; points on a sphere (w/ some concentration on center with shape of obj)
# sampled_cloud = Gen.random(T.uniform_mixture_from_template, gt_cloud, 0.0001, radius, bounds)
# # # Visualize that point cloud.
# # V.setup_visualizer()
# # V.viz(sampled_cloud)  # = V.viz(T.move_points_to_frame_b(c, camera_pose))

# +
# ########
# # camera downscaling
# ########
# using IJulia

# final = 24
# for downscale_factor=1:3.0:final
#     println("scale down by $downscale_factor")
    
#     # scale camera according to resolution; reset renderer
#     scaled_camera = GL.scale_down_camera(camera, downscale_factor)
        
#     # Set the renderer to now have those scaled down intrinsics.
#     GL.set_intrinsics!(renderer, scaled_camera)
 
#     d = GL.gl_render(renderer, [i], [pose], IDENTITY_POSE)

#     img = GL.view_depth_image(d)
#     img = I.imresize(img, (camera.height, camera.width));
#     IJulia.display(img)
  
#     # revert intrinsics
#     GL.set_intrinsics!(renderer, camera)

# end

# -

# # Intro demo of coarse-to-fine with particle filtering

# ## Particle Filtering + Orientation enumeration

# ### Orientation enumeration

function fibonacci_sphere(samples)
    points = []
    phi = π * (3. - sqrt(5.))
 
    for i in 0:(samples-1)
        y = 1 - (i / Float64(samples - 1)) * 2
        #@show y
        radius = sqrt(1 - y * y)

        theta = phi * i

        x = cos(theta) * radius
        z = sin(theta) * radius

        push!(points, (x, y, z))
    end
    return points
end

# +
unit_sphere_directions = fibonacci_sphere(50);
other_rotation_angle = collect(0:0.2:(2*π));

rotations_to_enumerate_over = [
    let
        T.geodesicHopf_select_axis(StaticArrays.SVector(dir), ang, 1)
    end
    for dir in unit_sphere_directions, 
        ang in other_rotation_angle
];
num_rotations_to_enumerate_over = length(rotations_to_enumerate_over);
# -

# ### Helpers

function viz_trace(trace)
    V.setup_visualizer()
#     V.reset_visualizer()
    V.viz(Gen.get_retval(trace).voxelized_cloud  ./ 10.0; color=I.colorant"red", channel_name=:gen);
    V.viz(Gen.get_retval(trace).obs_cloud ./ 10.0; color=I.colorant"blue", channel_name=:obs);
end

# Helper function to get point cloud from the object ids, object poses, and camera pose
# (see demo.jl)
function get_cloud(poses, ids, renderer, camera_pose)
#     @show poses, ids
    depth_image = GL.gl_render(renderer, ids, poses, camera_pose)
    cloud = GL.depth_image_to_point_cloud(depth_image, renderer.camera_intrinsics)
    if isnothing(cloud)
        cloud = zeros(3,1)
    else
        cloud = T.move_points_to_frame_b(cloud, camera_pose)
    end
    cloud
end

# +
bounds = (-100.0, 100.0, -100.0, 100.0,-100.0,300.0)
num_candidate_objs = 1

v_resolution = radius -> (radius * 0.5)  # distance between two points on the pointcloud 

"""sample point clouds at the given camera resolution and radius"""
Gen.@gen function model(radius, renderer)
    
    # fixed object (TODO: generalize to any obj id) and pose
    i = {:id} ~ Gen.categorical(fill(1.0/num_candidate_objs, (num_candidate_objs,)))
    p = {T.floating_pose_addr(1)} ~ T.uniformPose(-0.001, 0.001, -0.001, 0.001, 4.999, 5.001)
    
    gt_cloud = get_cloud([p], [i], renderer, camera_pose)    
    voxelized_cloud = GL.voxelize(gt_cloud, v_resolution(radius))
    obs_cloud = {T.obs_addr()} ~ T.uniform_mixture_from_template(voxelized_cloud, 0.0001, radius, bounds)
#     print("."); flush(stdout)
         
    (id=i, pose=p, ori=p.orientation, cloud=gt_cloud, voxelized_cloud=voxelized_cloud, rendered_clouds=[voxelized_cloud], obs_cloud=obs_cloud)

end
 
"""return scale down factor of camera given desired resolution"""
function scale_factor(resolution, final_resolution)
   return final_resolution-resolution+1 
end

# +
"""Selecting from a set of traces over all enumerated angles, generate a set of initial particles"""
Gen.@gen function generate_initial_pf_state(scored_traces, num_particles, U=Gen.DynamicDSLTrace{Gen.DynamicDSLFunction{Any}})
    # unpack traces
    traces = (t -> t[1]).(scored_traces)
    log_weights = (t -> t[2]).(scored_traces) #; log_weights = log_weights .- Gen.logsumexp(log_weights)  # ∝ likelihood
    
    _, norm_log_weights = Gen.normalize_weights(log_weights)       
    
    # sample initial set of particles
    selected_traces = Vector{Gen.DynamicDSLTrace{Gen.DynamicDSLFunction{Any}}}(undef, num_particles)
    selected_log_weights = Vector{Float64}(undef, num_particles)
    for i=1:num_particles
        trace_idx = {:t_idx => i} ~ Gen.categorical(exp.(norm_log_weights))   
        selected_traces[i] = traces[trace_idx]
        selected_log_weights[i] = log_weights[trace_idx]
    end
    
    println("Initial particle state computed"); flush(stdout)

#     # visualize pdf of initial weights (?)
#     plot(selected_log_weights, seriestype=:stephist, fmt = :png)
    
    # see intermediate results
    top_n = 10
    _, norm_log_weights = Gen.normalize_weights(selected_log_weights)
    p = sortperm(norm_log_weights)[end-top_n:end]; 

    best_trace = selected_traces[argmax(norm_log_weights)];
    viz_trace(best_trace);
    println("top$top_n traces:")
    for idx in p
       println("current weight=", norm_log_weights[idx], "\t orientation=", Gen.get_retval(selected_traces[idx]).ori[1:3]) 
    end
    flush(stdout)
    
    return Gen.ParticleFilterState{U}(selected_traces, Vector{U}(undef, num_particles), selected_log_weights, 0., collect(1:num_particles))

end

# +

function icp_move_no_unexplained(trace, i, inf_radius, renderer, cam_pose; iterations=10)
    # get_cloud_func needs to give the points in the world frame
    get_cloud_func = (poses, ids, cam_pose, i) -> get_cloud(poses, ids, renderer, cam_pose)  
    id = Gen.get_choices(trace)[:id]  
    addr = T.floating_pose_addr(i)
    
    obs_cloud = T.move_points_to_frame_b(T.get_obs_cloud(trace), cam_pose)
    
    refined_pose = trace[addr]
    refined_pose = T.icp_object_pose(
        refined_pose,
        obs_cloud,
        p -> T.voxelize(get_cloud_func([p], [id], cam_pose, 1),inf_radius)
    )

    acceptances = false
     
    for _ in 1:iterations
        trace, acc = T.pose_mixture_move(
            trace, addr, [trace[addr], refined_pose], [0.5, 0.5], 1e-2, 5000.0
        )
        acceptances = acc || acceptances
    end
    
    trace, acceptances, refined_pose
end

# +
"""adapted from gen pf_step:
Perform a particle filter update, where the model arguments are adjusted, new observations are added, and the default proposal is used for new latent state.
"""
function particle_filter_step!(state::Gen.ParticleFilterState{U}, new_args::Tuple, argdiffs::Tuple,
        observations) where {U}    
    radius, renderer = new_args
    log_incremental_weights = Vector{Float64}(undef, num_particles) 
    radius::Float64 = new_args[1]
    
    @showprogress for i=1:num_particles
    ## do mh, drift moves, etc. tune particle before update (i.e. new likelihood)
        @time state.traces[i], acc, _ = icp_move_no_unexplained(state.traces[i], Gen.get_retval(state.traces[i]).id, 
                                            radius, renderer, camera_pose; iterations=15)

#         state.traces[i], acc = T.drift_move(state.traces[i], T.floating_pose_addr(1), 0.001, 10.0)
#         state.traces[i], acc = T.drift_move(state.traces[i], T.floating_pose_addr(1), 0.001, 100.0)
#         state.traces[i], acc = T.drift_move(state.traces[i], T.floating_pose_addr(1), 0.001, 1000.0)

        
        # evolve the particle (with new radius involved in new_args)
        (state.new_traces[i], increment, _, discard) = Gen.update(
            state.traces[i], new_args, argdiffs, observations)
#         if !isempty(discard)
#             error("Choices were updated or deleted inside particle filter step: $discard")
#         end
        log_incremental_weights[i] = increment
        state.log_weights[i] += increment
    end
    
    # swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp

    return log_incremental_weights
end
# -

# ### control resolution with radius (fixed camera intrinsics)

# +
"""Particle filter on the evolution of agent knowledge over time; resolution control with radius"""


function particle_filter(renderer, init_radius::Float64, final_radius::Float64, enum_ori_traces,
                        gt_obj_id::Int, gt_pose::Pose, num_particles::Int, num_samples::Int, U=Gen.DynamicDSLTrace{Gen.DynamicDSLFunction{Any}})
    
    # initialize renderer and particle filter
    GL.set_intrinsics!(renderer, camera)  
    gt_cloud = get_cloud([gt_pose], [gt_obj_id], renderer, camera_pose)
    gt_voxelized_cloud = GL.voxelize(gt_cloud, v_resolution(init_radius))
#     V.setup_visualizer()
#     V.viz(gt_voxelized_cloud)
    println("\ninitializing particle filter at radius $init_radius") 
    flush(stdout)
    
    
    # initialize a state of initial particles with various rotations
    scored_traces = enum_ori_traces  # get precomputed
    
    state::Gen.ParticleFilterState{U} = generate_initial_pf_state(scored_traces, num_particles)
    
    
    # evolve over resolutions (modify sphere radius `r` of mixture point cloud likelihood)
    @assert(final_radius <= init_radius)
    step = -0.3 
    for radius in init_radius+step:step:final_radius
        println("\n========Radius=$radius========"); flush(stdout)
        Gen.maybe_resample!(state, ess_threshold=num_particles/3, verbose=true)  
        
        # update pf 
        observations = Gen.choicemap(T.obs_addr() => GL.voxelize(gt_cloud, v_resolution(radius)), :id => gt_obj_id)
        current_log_weights = particle_filter_step!(state, (radius, renderer), (Gen.UnknownChange(),), observations)    
        
        # see intermediate results
        top_n = 10
        _, norm_log_weights = Gen.normalize_weights(current_log_weights)
        p = sortperm(norm_log_weights)[end-top_n:end]; 
        
        best_trace = state.traces[argmax(norm_log_weights)];
        viz_trace(best_trace);
        println("top$top_n traces:")
        for idx in p
           println("current weight=", norm_log_weights[idx], "\t orientation=", Gen.get_retval(state.traces[idx]).ori[1:3]) 
        end
        flush(stdout)
        
    end;
    
     return Gen.sample_unweighted_traces(state, num_samples)
    
end

# +
# Initialize the canera intrinsics and renderer that will render using those intrinsics.
GL.set_intrinsics!(renderer, camera)
 
init_radius, final_radius = float(1.0), float(0.1)  # low to high "focus"
camera_pose = IDENTITY_POSE

gt_obj_id = 1   # cube
# gt_pose = T.Pose(pos=[0.0, 0.0, 5.0], orientation=R.QuatRotation(0.2, -0.5, 0.1, -0.8))
tr, _ = Gen.generate(model, (init_radius, renderer),                 
            Gen.choicemap(:id => 1))
gt_pose = T.Pose(pos=[0.0, 0.0, 5.0], orientation=Gen.get_retval(tr).ori)

function precompute_enumerations(gt_voxelized_cloud, gt_pose, init_radius, renderer, rotations_to_enumerate_over)
    # initialize a state of initial particles with various rotations
    println("Enumerating over all angles:")
    scored_traces = @showprogress map(orn -> 
                    Gen.generate(model, (init_radius, renderer),                 
                                Gen.choicemap(T.obs_addr() => gt_voxelized_cloud, 
                                            :id => 1, 
                                            T.floating_pose_addr(1) => Pose(gt_pose.pos, orn))),
                                rotations_to_enumerate_over[:]);   
    return scored_traces
end 
gt_cloud = get_cloud([gt_pose], [gt_obj_id], renderer, camera_pose)
gt_voxelized_cloud = GL.voxelize(gt_cloud, v_resolution(init_radius))
# precompute enumeration; run the below code once and comment out
enum_ori_traces = precompute_enumerations(gt_voxelized_cloud, gt_pose, init_radius, renderer, rotations_to_enumerate_over)

num_rotations_to_enumerate_over = length(enum_ori_traces)

# -

@show num_rotations_to_enumerate_over
num_particles = 50
num_samples = num_particles
@time pf_traces = particle_filter(renderer, init_radius, final_radius, enum_ori_traces, gt_obj_id, gt_pose, num_particles, num_samples);




# ### visualization of the enumerated posterior at multiple different resolutions

# xyz = [
#     rotations_to_enumerate_over[i,1] * [1, 0, 0]
#     for i in 1:size(rotations_to_enumerate_over)[1] 
# ];
# log_weights_xyz = zeros(size(rotations_to_enumerate_over)[1])
# weights_xyz = exp.(log_weights_xyz)
# order = sortperm(weights_xyz,rev=true)
# weights_xyz = weights_xyz[order]
# xyz = xyz[order]
# weights_xyz


# +
# using PyCall
# using PyPlot

# plt = pyimport("matplotlib.pyplot")
# mpl_toolkits = pyimport("mpl_toolkits")
# x = range(0;stop=2*pi,length=1000); y = sin.(3*x + 4*cos.(2*x));
# plot(x, y, color="red", linewidth=2.0, linestyle="--")
# # plt.show()

# function run_viz(x, y, z, c)
#     fig = plt.figure(figsize=(9, 6))
#     ax = plt.axes(projection="3d")
#     ax.scatter3D(x, y, z, c=c)
#     p = ax.set_title("3D scatterplot", pad=25, size=15)
#     ax.set_xlabel("X") 
#     ax.set_ylabel("Y") 
#     ax.set_zlabel("Z")
#     plt.show()
# end



# # PyCall.py"""
# # import matplotlib.pyplot as plt
# # from mpl_toolkits import mplot3d
# # def run_viz(x,y,z,c):
# #     fig = plt.figure(figsize=(9, 6))
# #     ax = plt.axes(projection='3d')
# #     ax.scatter3D(x, y, z, c=c)
# #     p = ax.set_title("3D scatterplot", pad=25, size=15)
# #     ax.set_xlabel("X") 
# #     ax.set_ylabel("Y") 
# #     ax.set_zlabel("Z")
# #     plt.show()
# # """

# run_viz( (i->i[1]).(xyz), (i->i[2]).(xyz), (i->i[3]).(xyz), weights_xyz)
# -


