# This is an absolutely disgusting notebook.
# Don't take anything in here (in terms of reusability) seriously.
# Most of this will be pulled apart.

using Revise
using Random
Random.seed!(314159)
using GLRenderer
import Images as I
import ImageView as IV
import MiniGSG as S
import Rotations as R
using PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
using NearestNeighbors
import LightGraphs as LG
import GenDirectionalStats as GDS
using Gen
using Dates
using UnicodePlots
using StatsBase
using Rotations
using GenInferenceDiagnostics
using GenSMCGF
using GenAIDE
using FileIO
using CairoMakie
import MeshCatViz as V

resolution = 1.0
box = GLRenderer.box_mesh_from_dims(ones(3));
occluder = GLRenderer.box_mesh_from_dims([2.0, 5.0, 0.1]);
occluder_pose = Pose([0.0, 0.0, 10.0], IDENTITY_ORN)
intrinsics = GLRenderer.CameraIntrinsics();
intrinsics = GLRenderer.scale_down_camera(intrinsics, 6)
renderer = GLRenderer.setup_renderer(intrinsics, GLRenderer.DepthMode())
GLRenderer.load_object!(renderer, occluder);
GLRenderer.load_object!(renderer, box);

#####
##### Model
#####

Gen.@gen function kernel(t, null, 
        positions, rotations, poses, 
        depth_images, voxelized_clouds, obs_clouds)
    if t == 1
        x ~ Gen.uniform(-6.0, 6.0)
        rot ~ GDS.uniform_rot3()
    else
        x ~ Gen.normal(positions[end], 0.8)
        rot ~ GDS.vmf_rot3(rotations[end], 300.0)
    end
    push!(positions, x)
    push!(rotations, rot)
    object_pose = Pose([x, 0.0, 12.0], rot)
    depth_image = GLRenderer.gl_render(renderer, [1, 2], 
                                       [occluder_pose, object_pose], 
                                       IDENTITY_POSE)
    rendered_cloud = GLRenderer.depth_image_to_point_cloud(depth_image, 
                                                           intrinsics)
    voxelized_cloud = GLRenderer.voxelize(rendered_cloud, resolution)
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

Gen.@gen function model(t)
    positions = []
    rotations = []
    poses = []
    depth_images = []
    voxelized_clouds = []
    obs_clouds = []
    m ~ unf(t, 0, positions, rotations,
            poses, depth_images, 
            voxelized_clouds, obs_clouds)
    return (; poses, depth_images, voxelized_clouds, obs_clouds)
end

#####
##### Inference with enumerative proposal
#####

mixture_of_normals = HomogeneousMixture(normal, [0, 0])

gold_mixture_ref = Ref(Dict{Int, Any}())
target_mixture_ref = Ref(Dict{Int, Any}())

# An enumerative proposal over linear translations.
Gen.@gen function initial_proposal(grid_step, chm, 
        mixture_ref::Ref)
    x_gridded = -5.0 : grid_step : 5.0
    weights = map(x_gridded) do x
        choices = choicemap((:m => 1 => :obs_cloud, 
                             chm[:m => 1 => :obs_cloud]),
                            (:m => 1 => :x, x))
        tr, w = generate(model, (1, ), choices)
        w
    end
    nw = exp.(weights .- Gen.logsumexp(weights))
    μ = sum(x_gridded .* nw)
    covariance = 0.3
    mixture_ref[][1] = (μ, covariance)
    {:m => 1 => :x} ~ normal(μ, covariance)
end

# An enumerative proposal over linear translations in local grid.
Gen.@gen function transition_proposal(trace, t, 
        grid_step, grid_radius, chm,
        mixture_ref::Ref)
    prev_chm = get_choices(trace)
    prev = get_choices(trace)[:m => t - 1 => :x]
    x_gridded = (prev - grid_radius) : grid_step : (prev + grid_radius)
    weights = map(x_gridded) do x
        choices = choicemap((:m => 1 => :obs_cloud,
                             chm[:m => t => :obs_cloud]),
                            (:m => 1 => :x, x))
        tr, w  = generate(model, (1, ), choices)
        w
    end
    nw = exp.(weights .- Gen.logsumexp(weights))
    μ = sum(x_gridded .* nw)
    covariance = 0.3
    mixture_ref[][t] = (μ, covariance)
    {:m => t => :x} ~ normal(μ, covariance)
end

smc_gf = GenSMCGF.SMCGF(model, 
                        initial_proposal, 
                        transition_proposal)


struct SMCSpec
    n_particles::Int
    grid_step
    grid_radius
end

#####
#####
#####

function infer(num_timesteps, obs, mixture_ref; 
        grid_step = 0.1, grid_radius = 3.0,
        N_particles = 5)
    chms = map(1 : num_timesteps) do t
        addr = :m => t => :obs_cloud
        choicemap((addr, obs[addr]))
    end
    proposal_args = Tuple[(t, grid_step, grid_radius, 
                           obs, mixture_ref) for t in 2 : num_timesteps]
    pushfirst!(proposal_args, (grid_step, obs, mixture_ref))
    args = [(t, ) for t in 1 : num_timesteps]
    argdiffs = [(Gen.IntDiff(1), ) for t in 1 : num_timesteps]
    smc_tr = simulate(smc_gf, (chms, args, argdiffs, 
                               proposal_args, N_particles))
    particles = smc_tr.population
    log_weights = smc_tr.log_weights
    log_total_weight = Gen.logsumexp(log_weights)
    normalized_weights = exp.(log_weights .- log_total_weight)
    rets = [particles[categorical(normalized_weights)] for _ in 1 : length(particles)]
    return rets
end

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

function run_msbc(N, N_inf, N_particles,
        num_timesteps, grid_step, grid_radius)
    population_msbc = [ simulate(model, (num_timesteps,)) for _ in 1 : N]
    probe_msbc = Gen.select((:m => i => :x for i in 1 : num_timesteps)...)
    obs_msbc = Gen.select((:m => i => :obs_cloud for i in 1 : num_timesteps)...)
    function inference_msbc(obs)
        chms = map(1 : num_timesteps) do t
            addr = :m => t => :obs_cloud
            choicemap((addr, obs[addr]))
        end
        proposal_args = Tuple[(t, grid_step, grid_radius, 
                               obs, gold_mixture_ref) for t in 2 : num_timesteps]
        pushfirst!(proposal_args, (grid_step, obs, gold_mixture_ref))
        args = [(t, ) for t in 1 : num_timesteps]
        argdiffs = [(Gen.IntDiff(1), ) for t in 1 : num_timesteps]
        rets = []
        for k in 1 : N_inf
            smc_tr = simulate(smc_gf, (chms, args, argdiffs, 
                                       proposal_args, N_particles))
            push!(rets, smc_tr.chosen_particle)
        end
        return rets
    end
    summary = GenInferenceDiagnostics.msbc(population_msbc,
                                           probe_msbc,
                                           obs_msbc,
                                           inference_msbc; 
                                           monitor = true)
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

#####
##### Visualizer
#####

function animate_traces_with_gif!(gif, timesteps::Int, 
        population::Vector, name::String)
    fig = Figure(resolution = (1300, 1300), fontsize = 34)
    ax_mix_hist = Axis(fig[2, 1], ylabel = "proposal density", 
                       labelsize = 48)
    hidexdecorations!(ax_mix_hist)
    xlims!(ax_mix_hist, (-5.0, 5.0))
    ylims!(ax_mix_hist, (0.0, 3.0))
    ax = Axis(fig[2, 1], xlabel = "x", ylabel = "y")
    xlims!(ax, (-5.0, 5.0))
    ylims!(ax, (0.0, 10.0))
    ax_img = Axis(fig[1, 1])
    hidedecorations!(ax)
    xlims!(ax_img, (-5.0, 5.0))
    ylims!(ax_img, (0.0, 10.0))
    hidedecorations!(ax_img)
    particle_clouds = map(1 : timesteps) do t
        map(population) do tr
            chm = get_choices(tr)
            (chm[:m => t => :x], 5.0)
        end
    end
    time = Observable(1)
    ps = @lift(particle_clouds[$time])
    img = @lift(rotr90(gif[:, :, $time]))
    gold_mix_hist = Observable(Float64[1.0])
    hist!(ax_mix_hist, gold_mix_hist; bins = 30, 
          normalization = :pdf, color = :gold4)
    image!(ax_img, img)
    scatter!(ax, ps; markersize = 14, color = :gold4)
    record(fig, name, 1 : timesteps) do t
        gold_mix_hist[] = [normal(gold_mixture_ref[][t]...) 
                           for _ in 1 : 10000]
        time[] = t
        notify(ps)
        notify(img)
    end
end

#####
##### KLISH
#####

function flatten_to_ticks(v::Vector{SMCSpec})
    map(v) do spec
        repr((spec.n_particles, spec.grid_step))
    end
end

function klish(T::Int, obs::ChoiceMap,
        gold_standard::SMCSpec,
        search_schedule::Vector{SMCSpec};
        name = "")

    ground_truth_x = map(1 : T) do t
        (chm[:m => t => :x], 5.0)
    end

    # Setup gold standard.
    gold_grid_step = gold_standard.grid_step
    gold_grid_radius = gold_standard.grid_radius
    gold_n_particles = gold_standard.n_particles
    chms = map(1 : T) do t
        addr = :m => t => :obs_cloud
        choicemap((addr, obs[addr]))
    end
    gold_proposal_args = Tuple[(t, gold_grid_step, gold_grid_radius,
                                obs, gold_mixture_ref)
                               for t in 2 : T]
    pushfirst!(gold_proposal_args, (gold_grid_step, obs, 
                                    gold_mixture_ref))
    args = [(t, ) for t in 1 : T]
    argdiffs = [(Gen.IntDiff(1), ) for t in 1 : T]

    # Setup animation.
    fig = Figure(resolution = (1300, 1300), fontsize = 34)
    ax_mix_hist = Axis(fig[2, 1], ylabel = "proposal density", 
                       labelsize=40)
    hidexdecorations!(ax_mix_hist)
    xlims!(ax_mix_hist, (-5.0, 5.0))
    ylims!(ax_mix_hist, (0.0, 3.0))
    ax = Axis(fig[2, 1], xlabel = "x")
    xlims!(ax, (-5.0, 5.0))
    ylims!(ax, (0.0, 10.0))
    ax_img = Axis(fig[1, 1])
    hidedecorations!(ax)
    xlims!(ax_img, (-5.0, 5.0))
    ylims!(ax_img, (0.0, 10.0))
    hidedecorations!(ax_img)
    ax_aide = Axis(fig[:, 2], xlabel = "search sequence", 
                   ylabel = "(scaled) AIDE estimate", labelsize=34)
    xlims!(ax_aide, (0, length(search_schedule) + 1))
    ax_aide.xticks = (1 : length(search_schedule), 
                      flatten_to_ticks(search_schedule))

    time = Observable(1)
    img = @lift(rotr90(gif[:, :, $time]))
    aide_estimates = Observable(Tuple{Int, Float64}[])
    target_mix_hist = Observable(Float64[1.0])
    gold_mix_hist = Observable(Float64[1.0])
    hist!(ax_mix_hist, target_mix_hist; bins = 30, 
          normalization = :pdf, color = :blue)
    hist!(ax_mix_hist, gold_mix_hist; bins = 30, 
          normalization = :pdf, color = :gold4)
    image!(ax_img, img)
    ps = Ref(Vector{Tuple{Float64, Float64}}[[(0.0, 5.0)]])
    particle_clouds = @lift(ps[][$time])
    gt_particle = @lift(ground_truth_x[$time])
    scatter!(ax, particle_clouds; markersize = 14, color = :blue)
    scatter!(ax, gt_particle; marker = :star5,
             markersize = 17, color = :black)
    scatter!(ax_aide, aide_estimates; markersize = 14)
    steps = length(search_schedule)

    # Get gold traces.
    @time traces = infer(l, obs, gold_mixture_ref; 
                         N_particles = gold_n_particles, 
                         grid_step = gold_grid_step)

    gold_particles = map(1 : T) do t
        map(traces) do tr
            chm = get_choices(tr)
            (chm[:m => t => :x], 5.0)
        end
    end
    gold_particle_clouds = @lift(gold_particles[$time])
    scatter!(ax, gold_particle_clouds; markersize = 14, color = :gold4)

    # Iteratively tune.
    estimates = Dict{SMCSpec, Float64}()
    particles = Dict{Int, Any}()
    search_time_grid = Iterators.product(1 : T, 1 : steps)

    # Parallel compute AIDE estimates.
    search_prod = map(collect(Iterators.product([(T, gold_standard, )], search_schedule))) do t
        (t[1]..., t[2])
    end

    @time estimates = Dict(map(run_aide, search_prod))

    est_min = minimum(values(estimates))
    est_max = maximum(values(estimates))
    for (k, v) in estimates
        estimates[k] = (v - est_min) / (est_max - est_min)
    end
    ylims!(ax_aide, (0.0, 1.2))

    record(fig, name, search_time_grid) do (t, step)
        s = search_schedule[step]
        if haskey(particles, step)
            ps[] = particles[step]
        else
            target_grid_step = s.grid_step
            target_grid_radius = s.grid_radius
            target_n_particles = s.n_particles
            @time trs = infer(T, obs, target_mixture_ref; 
                              N_particles = target_n_particles, 
                              grid_radius = target_grid_radius,
                              grid_step = target_grid_step)
            particles[step] = map(1 : T) do t
                map(trs) do tr
                    chm = get_choices(tr)
                    (chm[:m => t => :x], 5.0)
                end
            end
            ps[] = particles[step]
        end

        estimate = estimates[s]
        push!(aide_estimates[], (step, estimate))
        gold_mix_hist[] = [normal(gold_mixture_ref[][t]...) for _ in 1 : 10000]
        target_mix_hist[] = [normal(target_mixture_ref[][t]...) for _ in 1 : 10000]

        # Update animations.
        time[] = t
        notify(gt_particle)
        notify(particle_clouds)
        notify(gold_particle_clouds)
        notify(aide_estimates)
        notify(target_mix_hist)
        notify(gold_mix_hist)
        notify(img)
    end
    return estimates
end

#####
##### Generate ground truth
#####

label = now()

# Ground truth.
range = collect(-5.0 : 0.5 : 5.0)
range = map(range) do r
    normal(r, 0.2)
end
l = length(range)
chm = choicemap(((:m => i => :x, x) for (i, x) in enumerate(range))...)
trace, = generate(model, (l,), chm);
depth_images = get_retval(trace).depth_images
x = GLRenderer.view_depth_image.(depth_images);
gif = cat(GLRenderer.view_depth_image.(depth_images)...; dims=3);

# Get ground truth.
obs = get_choices(trace)

#####
##### Visuals
#####

function run_aide(arg::Tuple{Int, SMCSpec, SMCSpec})
    T, gold_standard, s = arg
    aide_iters = 50
    mq = 10
    gold_grid_step = gold_standard.grid_step
    gold_grid_radius = gold_standard.grid_radius
    gold_n_particles = gold_standard.n_particles
    chms = map(1 : T) do t
        addr = :m => t => :obs_cloud
        choicemap((addr, obs[addr]))
    end
    gold_proposal_args = Tuple[(t, gold_grid_step, gold_grid_radius,
                                obs, target_mixture_ref)
                               for t in 2 : T]
    pushfirst!(gold_proposal_args, (gold_grid_step, obs, 
                                    target_mixture_ref))
    args = [(t, ) for t in 1 : T]
    argdiffs = [(Gen.IntDiff(1), ) for t in 1 : T]
    target_grid_step = s.grid_step
    target_grid_radius = s.grid_radius
    target_n_particles = s.n_particles
    target_proposal_args = Tuple[(t, target_grid_step, 
                                  target_grid_radius,
                                  obs, target_mixture_ref)
                                 for t in 2 : T]
    pushfirst!(target_proposal_args, (target_grid_step, obs, 
                                      target_mixture_ref))
    estimate, _ = aide(smc_gf, (chms, args, argdiffs, 
                                gold_proposal_args, gold_n_particles),
                       smc_gf, (chms, args, argdiffs, 
                                target_proposal_args, target_n_particles);
                       n = aide_iters,
                       mp = mq,
                       mq = mq)
    return s => estimate
end

# Diagnose smoothness of point cloud likelihood
range = collect(-5.0 : 0.6 : 5.0)
obs = get_choices(trace)
chms = map(range) do x
    obs_addr = :m => 1 => :obs_cloud
    choicemap((:m => 1 => :x, x),
              (obs_addr, obs[obs_addr]))
end

weights = map(chms) do chm
    tr, w = generate(model, (1, ), chm)
    w
end
normalized_weights = exp.(weights .- Gen.logsumexp(weights))
display(UnicodePlots.barplot(collect(range), normalized_weights))

mkdir("anims/$(label)")

# Animation
@info "Tracking animations."
for k in 1 : 8
    @time particles = infer(l, obs, gold_mixture_ref; 
                            N_particles = 1, 
                            grid_radius = 3.0,
                            grid_step = 0.2)
    animate_traces_with_gif!(gif, l, particles, 
                             "anims/$(label)/1d_occluder_particles_$k.gif")
end

# KLISH
gold_standard = SMCSpec(1, 0.2, 3.0)
search = [SMCSpec(1, 0.6, 1.8), 
          SMCSpec(1, 0.4, 1.2),
          SMCSpec(1, 0.6, 1.2)]

@info "KLISH animation."
estimates = klish(l, obs, 
                  gold_standard, search; 
                  name = "anims/$(label)/klish.gif")

@info "MSBC"
run_msbc(500, 40, 1, l, 0.2, 3.0)
