
isTree(g::LG.AbstractGraph) = LG.nv(g) == LG.ne(g)+1 && LG.is_connected(g)
isFloating(diforest::LG.SimpleDiGraph, i::Int) = i ∈ S.rootsOfForest(diforest)

export isTree, isFloating

import Base.parent
function parent(diforest::LG.SimpleDiGraph, i::Int)
    # Forests guarantee all vertices only have a single incoming edge.
    parents = LightGraphs.inneighbors(diforest, i)
    @assert length(parents) <= 1
    return isempty(parents) ? nothing : first(parents)
end

function children(diforest::LG.SimpleDiGraph, i::Int)
  # Forests guarantee all vertices only have a single incoming edge.
  children = LightGraphs.outneighbors(diforest, i)
  return children
end

export parent, children


# Construct directed graph with n vertices and no edges.
function no_edge_graph(n::Int)::LG.SimpleDiGraph
    g = LG.SimpleDiGraph()
    LG.add_vertices!(g, n)
    g
end

# Construct directed graph with n vertices and edges specfied as a list of (parent, child) tuples.
function graph_with_edges(n::Int, edges)::LG.SimpleDiGraph
    g = LG.SimpleDiGraph()
    LG.add_vertices!(g, n)
    for (i,j) in edges
        LG.add_edge!(g, i, j)
    end
    g
end

export no_edge_graph, graph_with_edges

function getDownstream(tree::LG.SimpleGraph, e::LG.Edge) :: Set{Int}
    nodes = Set{Int}()
    _getDownstream!(nodes, tree, e)
    nodes
end

function _getDownstream!(nodes::Set{Int}, tree::LG.SimpleGraph, e::LG.Edge)
    i, j = LG.src(e), LG.dst(e)
    if !((j in LG.neighbors(tree, i)) && (i in LG.neighbors(tree, j)))
        error("No edge (i, j)")
    end
    push!(nodes, j)
    for n in LG.neighbors(tree, j)
        if n != i
        _getDownstream!(nodes, tree, LG.Edge(j, n))
        end
    end
end

# Delete the edge from i->j and add an edge  from i->k.
function replaceEdge(tree::LG.SimpleGraph, i::Int, j::Int, k::Int)
    @assert LG.has_edge(tree, LG.Edge(i, j))
    @assert k in getDownstream(tree, LG.Edge(i, j))
    newTree = deepcopy(tree)
    LG.rem_edge!(newTree, i, j)
    LG.add_edge!(newTree, i, k)
    newTree
end

# Peform a BFS from a specified node, to assign directions to each of the edges.
# Then, remove that specified node from the directed graph and return what is left behind.
function decapitate(tree::LG.SimpleGraph; root::Union{Int,Nothing}=nothing)
    root = isnothing(root) ? LG.nv(tree) : root
    diforest = LG.bfs_tree(tree, root)
    LG.rem_vertex!(diforest, root)
    diforest
end

# Peform a BFS from a specified node, to assign directions to each of the edges.
# Then, remove that specified node from the directed graph and return what is left behind.
function decapitate(diforest::LG.SimpleDiGraph; root::Union{Int,Nothing}=nothing)
    root = isnothing(root) ? LG.nv(tree) : root
    LG.rem_vertex!(diforest, root)
    diforest
end

# Add a new node to a directed forest. Then, for each existing root node of the forest, add
# an edge from the newly added node to that root. This turns the forest into a tree.
function recapitate(diforest::LG.SimpleDiGraph)
    forestRoots = S.rootsOfForest(diforest)
    tree = LG.SimpleGraph(diforest)
    LG.add_vertex!(tree)
    root = LG.nv(tree)
    for forestRoot in forestRoots
        LG.add_edge!(tree, root, forestRoot)
    end
    @assert LG.ne(tree) == LG.nv(tree) - 1
    tree
end

function edges(g::LG.SimpleDiGraph)
    collect(LG.edges(g))
end



# Visualize Graph
function render_graph(g_in::LG.SimpleDiGraph; names=nothing, colors=nothing,title=nothing)

    num_verts = LG.nv(g_in)
    
    if isnothing(names)
        names = string.(collect(1:num_verts))
    end
    if isnothing(colors)
        colors = fill(I.colorant"tan", num_verts)
    end
    colors = map(x -> convert(I.RGBA,x), colors)

    graphviz = PyCall.pyimport("graphviz")
    g_out = graphviz.Digraph()
    g_out.attr("node", style="filled")
    for (i,v) in enumerate(LG.vertices(g_in))
      g_out.node(string(v), names[i], fillcolor="#"*I.hex(colors[i], :rrggbbaa))
    end
    for e in LG.edges(g_in)
      g_out.edge(string(LG.src(e)),
                 string(LG.dst(e)))
    end
    g_out


    max_width_px = 2000
    max_height_px = 2000
    dpi = 500
    g_out.attr("graph",
               # See https://graphviz.gitlab.io/_pages/doc/info/attrs.html#a:size
               size=string(max_width_px / dpi, ",",
                           max_height_px / dpi,
                           # Scale up if drawing is smaller than this size
                           "!"),
               dpi=string(dpi))
    if isnothing(title)
        g_out.attr(label=title)
    end
    g_out.render("/tmp/g", format="png")

    FileIO.load("/tmp/g.png")
end