using Revise
using Distributed
process_manager = Distributed.LocalManager(6, true)
procs = addprocs(process_manager; exeflags="--project")

@everywhere begin
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

    resolution = 0.8
    box = GLRenderer.box_mesh_from_dims(ones(3));
    occluder = GLRenderer.box_mesh_from_dims([2.0, 5.0, 0.1]);
    occluder_pose = Pose([0.0, 0.0, 10.0], IDENTITY_ORN)
    intrinsics = GLRenderer.CameraIntrinsics();
    intrinsics = GLRenderer.scale_down_camera(intrinsics, 7)
    renderer = GLRenderer.setup_renderer(intrinsics, GLRenderer.DepthMode())
    GLRenderer.load_object!(renderer, occluder);
    GLRenderer.load_object!(renderer, box);
end

#####
##### Model
#####

@everywhere Gen.@gen function kernel(t, null, 
        positions, rotations, poses, 
        depth_images, voxelized_clouds, obs_clouds)
    if t == 1
        x ~ Gen.uniform(-6.0, 6.0)
        rot ~ GDS.uniform_rot3()
    else
        x ~ Gen.normal(positions[end], 0.5)
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

@everywhere unf = Unfold(kernel)

@everywhere Gen.@gen function model(t)
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

@everywhere Gen.@load_generated_functions()

#####
##### Inference with enumerative proposal
#####

@everywhere mixture_of_normals = HomogeneousMixture(normal, [0, 0])

# An enumerative proposal over linear translations.
@everywhere Gen.@gen function initial_proposal(grid_step, chm)
    x_gridded = -5.0 : grid_step : -3.0
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
@everywhere Gen.@gen function transition_proposal(trace, t, grid_step, chm)
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

@everywhere Gen.@gen function rejuv_proposal(tr, t::Int)
    x_cur = tr[:m => t => :x]
    {:m => t => :x} ~ normal(x_cur, 0.2)
end

@everywhere Gen.@kern function k1(tr, t::Int)
    tr ~ mh(tr, rejuv_proposal, (t, ))
end

@everywhere smc_gf = GenSMCGF.SMCGF(model, 
                                    initial_proposal, transition_proposal, 
                                    k1)

@everywhere struct SMCSpec
    n_particles::Int
    n_rejuv::Int
    grid_step
end

#####
#####
#####

function infer(num_timesteps, obs; 
        grid_step = 0.1, N_particles = 5, N_rejuvenation = 1)
    smc_gf = GenSMCGF.SMCGF(model, 
                            initial_proposal, transition_proposal, 
                            k1)
    chms = map(1 : num_timesteps) do t
        addr = :m => t => :obs_cloud
        choicemap((addr, obs[addr]))
    end
    proposal_args = Tuple[(t, grid_step, obs) for t in 2 : num_timesteps]
    pushfirst!(proposal_args, (0.02, obs))
    args = [(t, ) for t in 1 : num_timesteps]
    argdiffs = [(Gen.IntDiff(1), ) for t in 1 : num_timesteps]
    smc_tr = simulate(smc_gf, (chms, args, argdiffs, 
                               proposal_args, args, 
                               N_particles, N_rejuvenation))
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

function run_msbc(N, N_inf, N_particles, N_rejuvenation, 
        num_timesteps, grid_step)
    population_msbc = [ simulate(model, (num_timesteps,)) for _ in 1 : N]
    probe_msbc = Gen.select((:m => i => :x for i in 1 : num_timesteps)...)
    obs_msbc = Gen.select((:m => i => :obs_cloud for i in 1 : num_timesteps)...)
    function inference_msbc(obs)
        smc_gf = GenSMCGF.SMCGF(model, 
                                initial_proposal, transition_proposal, 
                                k1)
        chms = map(1 : num_timesteps) do t
            addr = :m => t => :obs_cloud
            choicemap((addr, obs[addr]))
        end
        proposal_args = Tuple[(t, grid_step, obs) for t in 2 : num_timesteps]
        pushfirst!(proposal_args, (grid_step, obs))
        args = [(t, ) for t in 1 : num_timesteps]
        argdiffs = [(Gen.IntDiff(1), ) for t in 1 : num_timesteps]
        smc_tr = simulate(smc_gf, (chms, args, argdiffs, 
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
    fig = Figure(resolution = (1600, 1600))
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
    image!(ax_img, img)
    scatter!(ax, ps; markersize = 6)
    record(fig, name, 1 : timesteps) do t
        time[] = t
        notify(particle_clouds)
        notify(img)
    end
end

#####
##### KLISH
#####

function flatten_to_ticks(v::Vector{SMCSpec})
    map(v) do spec
        repr((spec.n_particles, spec.n_rejuv, spec.grid_step))
    end
end

function klish(T::Int, obs::ChoiceMap,
        gold_standard::SMCSpec,
        search_schedule::Vector{SMCSpec};
        aide_iters = 2, mq = 1, name = "")
    smc_gf = GenSMCGF.SMCGF(model, 
                            initial_proposal, transition_proposal, 
                            k1)

    # Setup gold standard.
    gold_grid_step = gold_standard.grid_step
    gold_n_rejuv = gold_standard.n_rejuv
    gold_n_particles = gold_standard.n_particles
    chms = map(1 : T) do t
        addr = :m => t => :obs_cloud
        choicemap((addr, obs[addr]))
    end
    gold_proposal_args = Tuple[(t, gold_grid_step, obs) 
                               for t in 2 : T]
    pushfirst!(gold_proposal_args, (0.01, obs))
    args = [(t, ) for t in 1 : T]
    argdiffs = [(Gen.IntDiff(1), ) for t in 1 : T]

    # Setup animation.
    fig = Figure(resolution = (1600, 1600))
    ax = Axis(fig[2, 1], xlabel = "x", ylabel = "y")
    xlims!(ax, (-5.0, 5.0))
    ylims!(ax, (0.0, 10.0))
    ax_img = Axis(fig[1, 1])
    hidedecorations!(ax)
    xlims!(ax_img, (-5.0, 5.0))
    ylims!(ax_img, (0.0, 10.0))
    hidedecorations!(ax_img)
    ax_aide = Axis(fig[:, 2], xlabel = "search sequence", 
                   ylabel = "AIDE estimate")
    xlims!(ax_aide, (0, length(search_schedule)))
    ax_aide.xticks = (1 : length(search_schedule), 
                      flatten_to_ticks(search_schedule))
    ylims!(ax_aide, (-5.0, 10.0))

    time = Observable(1)
    img = @lift(rotr90(gif[:, :, $time]))
    aide_estimates = Observable(Tuple{Int, Float64}[])
    image!(ax_img, img)
    ps = Ref(Vector{Tuple{Float64, Float64}}[[(0.0, 5.0)]])
    particle_clouds = @lift(ps[][$time])
    scatter!(ax, particle_clouds; markersize = 6, color = :red)
    scatter!(ax_aide, aide_estimates)
    steps = length(search_schedule)

    @time traces = infer(l, obs; 
                         N_particles = gold_n_particles, 
                         N_rejuvenation = gold_n_rejuv, 
                         grid_step = gold_grid_step)
    gold_particles = map(1 : T) do t
        map(traces) do tr
            chm = get_choices(tr)
            (chm[:m => t => :x], 5.0)
        end
    end
    gold_particle_clouds = @lift(gold_particles[$time])
    scatter!(ax, gold_particle_clouds; markersize = 6, color = :blue)

    # Iteratively tune.
    estimates = Dict{SMCSpec, Float64}()
    particles = Dict{Int, Any}()
    search_time_grid = Iterators.product(1 : T, 1 : steps)

    # Parallel compute AIDE estimates.
    search_prod = map(collect(Iterators.product([(T, gold_standard, )], search_schedule))) do t
        (t[1]..., t[2])
    end
    estimates = Dict(pmap(run_aide, search_prod))

    record(fig, name, search_time_grid) do (t, step)
        s = search_schedule[step]
        if haskey(particles, step)
            ps[] = particles[step]
        else
            target_grid_step = s.grid_step
            target_n_particles = s.n_particles
            target_n_rejuv = s.n_rejuv
            target_proposal_args = Tuple[(t, target_grid_step, obs) 
                                         for t in 2 : T]
            pushfirst!(target_proposal_args, (0.01, obs))
            @time trs = infer(T, obs; 
                              N_particles = target_n_particles, 
                              N_rejuvenation = target_n_rejuv,
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

        # Update animations.
        time[] = t
        notify(particle_clouds)
        notify(gold_particle_clouds)
        notify(aide_estimates)
        notify(img)
    end
    return estimates
end

#####
##### Generate ground truth
#####

@everywhere begin
    label = now()

    # Ground truth.
    range = collect(-5.0 : 0.5 : -3.0)
    l = length(range)
    chm = choicemap(((:m => i => :x, x) for (i, x) in enumerate(range))...)
    trace, = generate(model, (l,), chm);
    depth_images = get_retval(trace).depth_images
    x = GLRenderer.view_depth_image.(depth_images);
    gif = cat(GLRenderer.view_depth_image.(depth_images)...; dims=3);

    # Get ground truth.
    obs = get_choices(trace)
end

#####
##### Visuals
#####

@everywhere function run_aide(arg::Tuple{Int, SMCSpec, SMCSpec})
    T, gold_standard, s = arg
    aide_iters = 1
    mq = 1
    gold_grid_step = gold_standard.grid_step
    gold_n_rejuv = gold_standard.n_rejuv
    gold_n_particles = gold_standard.n_particles
    chms = map(1 : T) do t
        addr = :m => t => :obs_cloud
        choicemap((addr, obs[addr]))
    end
    gold_proposal_args = Tuple[(t, gold_grid_step, obs) 
                               for t in 2 : T]
    pushfirst!(gold_proposal_args, (0.01, obs))
    args = [(t, ) for t in 1 : T]
    argdiffs = [(Gen.IntDiff(1), ) for t in 1 : T]
    target_grid_step = s.grid_step
    target_n_particles = s.n_particles
    target_n_rejuv = s.n_rejuv
    target_proposal_args = Tuple[(t, target_grid_step, obs) 
                                 for t in 2 : T]
    pushfirst!(target_proposal_args, (0.01, obs))
    estimate, _ = aide(smc_gf, (chms, args, argdiffs, 
                                gold_proposal_args, args, 
                                gold_n_particles, gold_n_rejuv),
                       smc_gf, (chms, args, argdiffs, 
                                target_proposal_args, args,
                                target_n_particles, target_n_rejuv);
                       n = aide_iters,
                       mp = mq, # Exact inference
                       mq = mq)
    return s => estimate
end

# Diagnose smoothness of point cloud likelihood
#range = collect(-5.0 : 0.1 : 3.0)
#obs = get_choices(trace)
#chms = map(range) do x
#    obs_addr = :m => 1 => :obs_cloud
#    choicemap((:m => 1 => :x, x),
#              (obs_addr, obs[obs_addr]))
#end
#
#weights = map(chms) do chm
#    tr, w = generate(model, (1, ), chm)
#    w
#end
#normalized_weights = exp.(weights .- Gen.logsumexp(weights))
#display(UnicodePlots.barplot(collect(range), normalized_weights))

mkdir("anims/$(label)")

# KLISH
gold_standard = SMCSpec(5, 5, 0.01)
search = [SMCSpec(5, 5, 0.05), 
          SMCSpec(5, 4, 0.05),
          SMCSpec(5, 3, 0.05),
          SMCSpec(5, 2, 0.1)]
estimates = klish(l, obs, gold_standard, search; 
                  aide_iters = 5, mq = 5,
                  name = "anims/$(label)/klish.gif")

# Animation
#@time particles = infer(l, obs; 
#                        N_particles = 50, N_rejuvenation = 3, 
#                        grid_step = 0.05)
#
#animate_traces_with_gif!(gif, l, particles, 
#                         "anims/$(label)/1d_occluder_particles.gif")
