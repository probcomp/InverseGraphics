import LightGraphs

function prufer_code_to_tree(p)
    n = length(p)

    deg = [sum(p .== i) + 1 for i=1:(n+2)]
    G = LG.SimpleGraph(n+2)
    for i in p
        for j=1:(n+2)
            if deg[j] == 1
                LG.add_edge!(G, i,j)
                deg[i] -= 1
                deg[j] -= 1
                break
            end
        end
    end

    u = 0
    v = 0
    for i=1:(n+2)
        if deg[i] == 1
            if u==0
                u = i
            else
                v = i
                break
            end
        end
    end
    LG.add_edge!(G,u,v)

    G
end

function get_all_undirected_trees(num_verts)
    possible_codes = collect(Base.Iterators.product([1:num_verts for _ in 1:(num_verts-2)]...))
    possible_graphs = prufer_code_to_tree.(possible_codes)[:]
end

function get_all_possible_scene_graphs(num_verts; depth_limit=nothing)
    possible_undirected_graphs = get_all_undirected_trees(num_verts)
    possible_scene_graphs = map(G -> LG.bfs_tree(G, num_verts), possible_undirected_graphs)
    if !isnothing(depth_limit)
        dists = maximum.(LightGraphs.gdistances.(possible_scene_graphs, LG.nv(possible_scene_graphs[1])));
        possible_scene_graphs = possible_scene_graphs[dists .<= depth_limit];
    end
    possible_scene_graphs
end

    
