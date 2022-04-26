using DataStructures

@dist labeled_categorical(labels, probs) = labels[Gen.categorical(probs)]

export labeled_categorical

Rot2(a::Real) = Rotations.RotMatrix{2,Float64}(a)

function floatingPosesOf(sg)
    fp = S.floatingPosesOf(sg)
    collect(values(fp))
end

POSE_DISTANCE_FUNC_RADIUS = 1.0

function euclidean_distance(pt1::SVector{3,<:Real}, pt2::SVector{3,<:Real})
    return sqrt(sum((x2 - x1)^2 for (x1, x2) in zip(pt1, pt2)))
end
function angle_distance(rot1::R.Rotation{3}, rot2::R.Rotation{3})
    q1 = PC.componentsWXYZ(Rotations.UnitQuaternion(rot1))
    q2 = PC.componentsWXYZ(Rotations.UnitQuaternion(rot2))
    return acos(min(1.0, 2 * sum(q1 .* q2)^2 - 1))
end
function pose_distance(p1::Pose, p2::Pose; r::Float64 = POSE_DISTANCE_FUNC_RADIUS)
    euclidean_distance(p1.pos, p2.pos) + r * angle_distance(p1.orientation, p2.orientation)
end

export Rot2, floatingPosesOf

function normalize_log_weights(log_weights)
    log_total_weight = Gen.logsumexp(log_weights)
    normalized_weights =  exp.(log_weights .- log_total_weight)
    normalized_weights
end

normalized_logprob(log_p) = exp.(log_p .- maximum(log_p)) ./ sum(exp.(log_p .- maximum(log_p)))

sample_n_from_categorical(weights, n) = [Gen.random(categorical,weights) for _ in 1:n]

export normalized_logprob, normalize_log_weights, sample_n_from_categorical

using Statistics

function discretize(cloud, resolution)
    round.(cloud ./ resolution) * resolution
end

function uniquify(cloud)
    idxs = unique(i -> cloud[:,i], 1:size(cloud)[2])
    cloud[:, idxs]
end

function voxelize(cloud, resolution)
    cloud = round.(cloud ./ resolution) * resolution
    idxs = unique(i -> cloud[:,i], 1:size(cloud)[2])
    cloud[:, idxs]
end

function min_max(cloud::Matrix{<:Real})
    minimum(cloud;dims=2), maximum(cloud;dims=2)
end

function min_max(cloud::Vector{<:Real})
    minimum(cloud), maximum(cloud)
end

function center_cloud(cloud)
    cloud .- mean(cloud, dims=2)
end

function opposite_face(face::Symbol)::Symbol
    pairs = [[:top, :bottom],[:left, :right],[:front, :back]]
    for p in pairs
        if face ∈ p
            idx = findfirst(p .== face)
            return p[((idx%2) +1)]
        end
    end
    return nothing
end

function get_unexplained_points(full_cloud, sub_cloud; radius=1.0)
    obs_tree = NearestNeighbors.KDTree(full_cloud)

    valid = fill(true, size(full_cloud)[2])
    idxs = NearestNeighbors.inrange(obs_tree, sub_cloud, radius)
    valid[unique(vcat(idxs...))] .= false
    valid
end

function rotation_between_two_vectors(u,v)
    u = u ./ LinearAlgebra.norm(u)
    v = v ./ LinearAlgebra.norm(v)
    half = (u+v) ./ LinearAlgebra.norm(u+v)
    q = R.QuatRotation(LinearAlgebra.dot(u,half),LinearAlgebra.cross(u,half)...)
end

function mix(a,b,alpha)
    alpha .* a .+ (1 - alpha) .* b
end

function camera_intrinsics_matrix(intrinsics::GL.CameraIntrinsics)
    K = zeros(3,3)
    K[1,1] = intrinsics.fx
    K[2,2] = intrinsics.fy
    K[1,3] = intrinsics.cx
    K[2,3] = intrinsics.cy
    K[3,3] = 1.0
    K
end

function fibonacci_sphere(samples)
    points = []
    phi = π * (3. - sqrt(5.))
    for i in 0:(samples-1)
        y = 1 - (i / Float64(samples - 1)) * 2
        radius = sqrt(1 - y * y)
        theta = phi * i
        x = cos(theta) * radius
        z = sin(theta) * radius
        push!(points, [x, y, z])
    end
    return points
end



export voxelize, min_max, discretize, center_cloud, pose_distance
