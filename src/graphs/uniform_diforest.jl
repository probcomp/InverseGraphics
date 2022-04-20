
# Uniform distribution over directed forests with n nodes.

struct UniformDiForest <: Distribution{LG.SimpleDiGraph} end
const uniform_diforest = UniformDiForest()
(::UniformDiForest)(n) = Gen.random(uniform_diforest, n)

# The number of directed forests on n nodes is (n+1) ^ (n-1) as given by Cayley's formula.
function Gen.logpdf(::UniformDiForest, diforest::LG.SimpleDiGraph, n::Int)
    n != LG.nv(diforest) && return -Inf
    - ((n - 1) * log(n+1))
end

function Gen.random(::UniformDiForest, num_verts::Int)
    prufer_code = [rand(1:(num_verts+1)) for _ in 1:(num_verts - 1)]
    g = T.prufer_code_to_tree(prufer_code)
    g = T.make_bfs_tree(g, 1)
    g = T.decapitate(g, 1)
end


export uniform_diforest
