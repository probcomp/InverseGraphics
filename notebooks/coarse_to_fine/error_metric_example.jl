import NearestNeighbors
import InverseGraphics as T

# !!! Replace this with object cloud. For YCB it might be id_to_cloud[id] !!!
obj_cloud = rand(3,5000) * 10.0

function error(object_cloud, ground_truth_pose, predicted_pose)
    tree = NearestNeighbors.KDTree(T.move_points_to_frame_b(object_cloud, predicted_pose))
    _, dists = NearestNeighbors.nn(tree, T.move_points_to_frame_b(object_cloud, ground_truth_pose))
    sum(dists) ./ length(dists)
end

p1 = T.Pose(ones(3), T.R.RotXYZ(0.1, 0.4, 0.2))

@time @show error(obj_cloud, p1, T.IDENTITY_POSE)
@time @show error(obj_cloud, T.IDENTITY_POSE, p1)
@time @show error(obj_cloud, p1, p1)
@time @show error(obj_cloud, T.IDENTITY_POSE, T.IDENTITY_POSE)
