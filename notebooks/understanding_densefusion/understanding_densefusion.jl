# -*- coding: utf-8 -*-
import Revise
import InverseGraphics as T
import InverseGraphics: Pose, IDENTITY_POSE, IDENTITY_ORN, GL, I, S
import Gen
import Open3DVisualizer as V
import NearestNeighbors as NN
import Statistics: mean

YCB_DIR = joinpath(dirname(dirname(pathof(T))),"data")
world_scaling_factor = 1.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)


id_to_xyz = T.load_ycbv_point_xyz_adjusted(YCB_DIR, world_scaling_factor, id_to_shift);

IDX = 200

# Load scene data.
#    gt_poses : Ground truth 6D poses of objects (in the camera frame)
#    ids      : object ids (order corresponds to the gt_poses list)
#    rgb_image, gt_depth_image :
#    cam_pose : 6D pose of camera (in world frame)
#    original_camera : Camera intrinsics
gt_poses, ids, gt_rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
df_poses, df_ids = T.load_ycbv_dense_fusion_predictions_adjusted(YCB_DIR, IDX, world_scaling_factor, id_to_shift)   

function compute_ADDS_error(xyz_models, gt_ids, gt_poses, pred_ids, pred_poses)
    add_s = []
    for (id,gt_pose) in zip(gt_ids, gt_poses)
        idx = findfirst(pred_ids .== id)
        if isnothing(idx)
            push!(add_s, 100.0)
            continue
        end
        pred_pose = pred_poses[idx]

        model = xyz_models[id]

        tree = NN.KDTree(T.GL.move_points_to_frame_b(model, pred_pose))
        _, dists = NN.nn(tree, T.GL.move_points_to_frame_b(model, gt_pose))

        push!(add_s, min(mean(dists)))
    end
    add_s
end

compute_ADDS_error(id_to_xyz, ids, gt_poses, df_ids, df_poses)

idx = 5
id = ids[idx]
gt_pose = gt_poses[idx]
df_idx = findfirst(df_ids .== ids[idx])
pred_pose = df_poses[df_idx]

model = id_to_xyz[id]
tree = NN.KDTree(T.GL.move_points_to_frame_b(model, gt_pose))
_, dists = NN.nn(tree, T.GL.move_points_to_frame_b(model, pred_pose))
add_s = mean(dists)

@show add_s

V.open_window()
V.make_point_cloud(T.move_points_to_frame_b(model, pred_pose); color=T.I.colorant"red")
V.make_point_cloud(T.move_points_to_frame_b(model, gt_pose); color=T.I.colorant"blue")

V.run()


