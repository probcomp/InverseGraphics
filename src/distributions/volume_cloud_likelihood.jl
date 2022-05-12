import Gen: geometric
import LinearAlgebra
import NearestNeighbors
import Memoize: @memoize
import StatsFuns: logsumexp

struct VolumeCloudLikelihood <: Distribution{Matrix{Float64}} end

const volume_cloud_likelihood = VolumeCloudLikelihood()

# Reference implementation of the sampler -- to clarify the semantics of the
# distribution, even if it's not used in code.
function Gen.random(::VolumeCloudLikelihood,
					X::Matrix{Float64},
                    p_outlier::Float64,
                    radius::Float64,
                    bounds::Tuple)
	X
end

@memoize function get_tree_from_cloud(Y)
    NearestNeighbors.KDTree(Y)
end

function Gen.logpdf(
        ::VolumeCloudLikelihood, Y::Matrix{Float64}, X::Matrix{Float64},
        p_outlier::Float64, radius::Float64, bounds::Tuple)
    size(X, 1) == 3 || error("X must have size 3 × something, each column represents a point.  Got size(X) = $(size(X))")
    size(Y, 1) == 3 || error("Y must have size 3 × something, each column represents a point.  Got size(Y) = $(size(Y))")
    all((Y .≥ [bounds[1], bounds[3], bounds[5]]) .&
        (Y .≤ [bounds[2], bounds[4], bounds[6]])
       ) || @warn("Some points in Y were out of bounds")

    tree = get_tree_from_cloud(Y)

    m = size(X, 2)
    n = size(Y, 2)
    logp_numpoints = Gen.logpdf(geometric, n,
                                1e-5)
    r = radius

    # all_idxs is an array of arrays, where the array all_idxs[i] contains all indices j such that
    # the points X[:,i] and Y[:,j] are within a distance of r units of each other.
    all_idxs = NearestNeighbors.inrange(tree, X, r)
    inside = fill(false, n)
    for idxs in all_idxs
        inside[idxs] .= true
    end

    num_inside = sum(inside)
    num_outside = n  - num_inside

    m_disc = size(voxelize(X, 2*r),2)
    Volume_inlier = 4.0 /3 * pi * r^3 * m_disc
    Volume_outlier = (bounds[2] - bounds[1]) * (bounds[4] - bounds[3]) * (bounds[6] - bounds[5])

    return (
        num_inside * log(p_outlier / Volume_outlier + (1 - p_outlier) / Volume_inlier) +
        num_outside * log(p_outlier / Volume_outlier)
    )
end
