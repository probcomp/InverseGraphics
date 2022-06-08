# gibbs sampling over discrete variables using enumeration

using Gen
using Plots

function cartesian_product(value_lists)
    tuples = Vector{Tuple}()
    for trace in value_lists[1]
        if length(value_lists) > 1
            append!(tuples,
                [(trace, rest...) for rest in cartesian_product(value_lists[2:end])])
        else
            append!(tuples, [(trace,)])
        end
    end
    return tuples
end

# Q: what if some of the variables aren't present in some of the enuemrated traces?
# A: it will return a 'did not visit all constraints error'

# Q: do you need to enumerate over all of the possible values of a variable?
# A: no, but if the current value is not one of the enumerated values (for
# discrete) or in one of the possible bins (for continuous), then it will
# reject

# Q: does it work with continuous variables too?
# A: yes, you provide a finite set of finite interval bins instead of a finite
# set of values

@gen function enumerative_gibbs_proposal(trace, spec, addrs, value_tuples, probabilities)
    idx ~ categorical(probabilities)
end

function get_cont_addr_bin(addrs_to_info, addr, index::Int)
    info = addrs_to_info[addr]
    bins = info[2]
    bin_start = bins[index]
    bin_end = bins[index+1]
    return (bin_start, bin_end)
end

function cont_bin_density(addrs, iscont, addrs_to_info, index_tuple)
    cont_score = 0.0
    for (j, addr) in enumerate(addrs)
        if iscont[j]
            (bin_start, bin_end) = get_cont_addr_bin(addrs_to_info, addr, index_tuple[j])
            cont_score -= log(bin_end - bin_start) # log 1/(bin length)
        end
    end
    return cont_score
end

function matches(trace, addrs, iscont, addrs_to_info, index_tuple)
    for j in 1:length(addrs)
        addr = addrs[j]
        if iscont[j]
            (bin_start, bin_end) = get_cont_addr_bin(addrs_to_info, addr, index_tuple[j])
            if !(bin_start <= trace[addr] < bin_end)
                return false
            end
        else
            value = addrs_to_info[addr][2][index_tuple[j]]
            if trace[addr] != value
                return false
            end
        end
    end
    return true
end

function enumerative_gibbs(trace::Trace, addrs_to_info::Dict)

    # addrs_to_info is a dictionary that maps addresses to tuple (:disc /
    # :cont, values / bins) NOTE: it only will work for scalar-valued
    # continuous variables for now.
        
    # construct cartesian product of test points
    addrs = Vector{Any}(undef, length(addrs_to_info))
    iscont = Vector{Bool}(undef, length(addrs_to_info))
    sizes = Vector{Any}(undef, length(addrs_to_info))
    cont_addrs = []
    for (i, (addr, info)) in enumerate(addrs_to_info)
        addrs[i] = addr
        if info[1] == :disc
            iscont[i] = false
            sizes[i] = collect(1:length(info[2]))
        elseif info[1] == :cont
            iscont[i] = true
            bins = info[2]
            #centers = bins[1:end-1] .+ diff(bins)/2 
            sizes[i] = collect(1:(length(bins)-1))
            push!(cont_addrs, addr)
        else
            error("invalid info for address $addr: $info")
        end
    end
    index_tuples = cartesian_product(sizes)

    # enumerate over test points and assess probability density at each
    # also identify which test point the current trace corresponds to (prev_idx)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    new_traces = Vector{Any}(undef, length(index_tuples))
    weights = Vector{Float64}(undef, length(index_tuples))
    prev_idx = -1
    for (i, index_tuple) in enumerate(index_tuples)

        # is this the test point corresponding to the previous trace?
        if prev_idx == -1 && matches(trace, addrs, iscont, addrs_to_info, index_tuple)
            prev_idx = i
        end

        # record new trace for this test point and its relative (log) probabilityS
        constraints = choicemap()
        for (addr, index) in zip(addrs, index_tuple)
            info = addrs_to_info[addr]
            if info[1] == :disc
                values = info[2]
                value = values[index]
            elseif info[1] == :cont
                bins = info[2]
                value = (bins[index] + bins[index+1])/2 # centerpoint
            else
                @assert false
            end
            constraints[addr] = value
        end
        (new_traces[i], weights[i], _, _) = update(trace, args, argdiffs, constraints)
    end

    # the current trace does not belong to any test point, the acceptance
    # probability is zero, because the probability of making the reverse move
    # is zero. reject.
    if prev_idx == -1
        return (trace, false)
    end

    # sample the new test point
    new_idx = categorical(exp.(weights .- logsumexp(weights)))
    new_trace = new_traces[new_idx]

    # sample values for each continuous random choice from bin uniform distributions
    # and compute forwards continuous proposal density
    cont_choices = choicemap()
    new_index_tuple = index_tuples[new_idx]
    for (j, addr) in enumerate(addrs)
        if iscont[j]
            (bin_start, bin_end) = get_cont_addr_bin(addrs_to_info, addr, new_index_tuple[j])
            cont_choices[addr] = uniform(bin_start, bin_end) # random sample from uniform distribution
        end
    end
    (new_trace, new_trace_weight, _, _) = update(new_trace, args, argdiffs, cont_choices)

    # compute MH acceptance probability
    backward_cont_score = cont_bin_density(addrs, iscont, addrs_to_info, index_tuples[prev_idx]) + weights[prev_idx]
    forward_cont_score = cont_bin_density(addrs, iscont, addrs_to_info, index_tuples[new_idx]) + weights[new_idx]
    alpha = (new_trace_weight + weights[new_idx]) + backward_cont_score - forward_cont_score

    # accept or reject
    if bernoulli(min(1.0, exp(alpha)))
        # accept
        return (new_trace, true)
    else
        # reject
        return (trace, false)
    end
end

##################################################
# test with discrete only (should always accept) #
##################################################

@gen function foo()
    b ~ bernoulli(0.5)
    if b
        a ~ categorical([0.25, 0.25, 0.25, 0.25])
    else
        a ~ categorical([0.4, 0.2, 0.2, 0.2])
    end
    c ~ normal(a, 1.0)
end

function always_accepts_test()
    for i in 1:100
        trace = simulate(foo, ())
        (_, accepted) = enumerative_gibbs(trace, Dict(
            :b => (:disc, [true, false]),
            :a => (:disc, [1, 2, 3, 4])))
        @assert accepted
    end
end

always_accepts_test()


##########################################
# test on a mostly deterministic problem #
##########################################

@gen function foo()
    a ~ uniform_discrete(1, 20)
    b ~ uniform_discrete(1, 20)
    c ~ uniform_discrete(1, 20)
    prob = (a == 10 && b == 14 && c == 3) ? 0.5 : 0.0000001
    x ~ bernoulli(prob)
end

function test_deterministic()
    trace, _ = generate(foo, (), choicemap((:x, true)))
    for i in 1:10
        @time (new_trace, accepted) = enumerative_gibbs(trace, Dict(
            :a => (:disc, collect(1:20)),
            :b => (:disc, collect(1:20)),
            :c => (:disc, collect(1:20))))
        @assert new_trace[:a] == 10
        @assert new_trace[:b] == 14
        @assert new_trace[:c] == 3
    end
end

#test_deterministic()

##################################################################
# test on a continuous problem (should look like Gibbs sampling) #
##################################################################

@gen function foo()
    x ~ normal(0, 1)
    y ~ normal(0, 1)
    z ~ normal(x + y, 0.01)
end

function test_continuous_problem()
    trace, _ = generate(foo, (), choicemap((:z, 2.0)))
    xs = [trace[:x]]
    ys = [trace[:y]]
    p1 = plot([], [], title="joint updates", label=nothing)
    for iter in 1:100
        @time (trace, accepted) = enumerative_gibbs(trace, Dict(
            :x => (:cont, collect(range(-3.0, 3.0, length=20))),
            :y => (:cont, collect(range(-3.0, 3.0, length=20)))))
        println(accepted)
        push!(xs, trace[:x])
        push!(ys, trace[:y])
        plot!([xs[iter], xs[iter+1]], [ys[iter], ys[iter+1]], label=nothing, color="black")
    end
    scatter!(xs, ys, label=nothing)
    xlims!(-3, 3)
    ylims!(-3, 3)

    trace, _ = generate(foo, (), choicemap((:z, 2.0)))
    xs = [trace[:x]]
    ys = [trace[:y]]
    p2 = plot([], [], title="single-variable updates", label=nothing)
    for iter in 1:100
        @time (trace, accepted) = enumerative_gibbs(trace, Dict(
            :x => (:cont, collect(range(-3.0, 3.0, length=20)))))
        println(accepted)
        @time (trace, accepted) = enumerative_gibbs(trace, Dict(
            :y => (:cont, collect(range(-3.0, 3.0, length=20)))))
        println(accepted)
        push!(xs, trace[:x])
        push!(ys, trace[:y])
        plot!([xs[iter], xs[iter+1]], [ys[iter], ys[iter+1]], label=nothing, color="black")
    end
    scatter!(xs, ys, label=nothing)
    xlims!(-3, 3)
    ylims!(-3, 3)

    plot(p1, p2)
    savefig("gridded_gibbs.png")
end

#test_continuous_problem()

######################################
# test on another continuous problem #
######################################

@gen function foo()
    x ~ normal(0, 1)
    y ~ normal(0, 1)
    r ~ normal(sqrt(x^2 + y^2), 0.01)
end

function test_continuous_problem2()
    trace, _ = generate(foo, (), choicemap((:r, 2.0)))
    xs = [trace[:x]]
    ys = [trace[:y]]
    p1 = plot([], [], title="joint updates", label=nothing)
    for iter in 1:100
        @time (trace, accepted) = enumerative_gibbs(trace, Dict(
            :x => (:cont, collect(range(-3.0, 3.0, length=20))),
            :y => (:cont, collect(range(-3.0, 3.0, length=20)))))
        println(accepted)
        push!(xs, trace[:x])
        push!(ys, trace[:y])
        plot!([xs[iter], xs[iter+1]], [ys[iter], ys[iter+1]], label=nothing, color="black")
    end
    scatter!(xs, ys, label=nothing)
    xlims!(-3, 3)
    ylims!(-3, 3)

    trace, _ = generate(foo, (), choicemap((:r, 2.0)))
    xs = [trace[:x]]
    ys = [trace[:y]]
    p2 = plot([], [], title="single-variable updates", label=nothing)
    for iter in 1:100
        @time (trace, accepted) = enumerative_gibbs(trace, Dict(
            :x => (:cont, collect(range(-3.0, 3.0, length=20)))))
        println(accepted)
        @time (trace, accepted) = enumerative_gibbs(trace, Dict(
            :y => (:cont, collect(range(-3.0, 3.0, length=20)))))
        println(accepted)
        push!(xs, trace[:x])
        push!(ys, trace[:y])
        plot!([xs[iter], xs[iter+1]], [ys[iter], ys[iter+1]], label=nothing, color="black")
    end
    scatter!(xs, ys, label=nothing)
    xlims!(-3, 3)
    ylims!(-3, 3)

    plot(p1, p2)
    savefig("gridded_gibbs_circle.png")
end

#test_continuous_problem2()


#########################################################
# test for model with discrete and continuous variables #
#########################################################

@gen function mixture_model()
    noise ~ inv_gamma(1, 1)
    mus = []
    for k in 1:4
        push!(mus, {(:mu, k)} ~ normal(0, 4))
    end
    for i in 1:10
        k = ({(:k, i)} ~ uniform_discrete(1, 4))
        {(:x, i)} ~ normal(mus[k], noise)
    end
end

function test_mixture()
    xs = [-3, -3, -3, 0, 0, 0, 3, 3, 3, 6]
    constraints = choicemap()
    for (i, x) in enumerate(xs)
        constraints[(:x, i)] = x
    end
    noise_bins = 10.0 .^ range(-3, 1, length=20)
    trace, _ = generate(mixture_model, (), constraints)
    traces = []
    for iter in 1:30
        println("iter: $iter")
        for k in 1:4
            for i in 1:10
                @time (trace, accepted) = enumerative_gibbs(trace, Dict(
                    (:k, i) => (:disc, collect(1:4)),
                    (:mu, k) => (:cont, collect(range(-8.0, 8.0, length=40))),
                    :noise => (:cont, noise_bins)))
                println(accepted)
                push!(traces, trace)
            end
        end
    end
    p1 = plot([], [], label=nothing)
    p2 = plot([], [], label=nothing)
    p3 = plot([trace[:noise] for trace in traces], label="noise")
    for k in 1:4
        plot!(p1, [trace[(:mu, k)] for trace in traces], label=nothing)
    end
    for i in 1:10
        plot!(p2, [trace[(:k, i)] for trace in traces], label=nothing)
    end
    plot(p1, p2, p3)
    savefig("mixture_model_trace.png")
end

test_mixture()


