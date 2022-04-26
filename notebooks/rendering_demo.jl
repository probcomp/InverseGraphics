# -*- coding: utf-8 -*-
import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import NearestNeighbors
import LightGraphs as LG
import Gen
import Plots
import FileIO
import ImageView as IV
using Measures
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end

V.setup_visualizer()

YCB_DIR = joinpath(dirname(dirname(pathof(T))),"data")
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)

IDX = 900
# Load scene data.
#    gt_poses : Ground truth 6D poses of objects (in the camera frame)
#    ids      : object ids (order corresponds to the gt_poses list)
#    rgb_image, gt_depth_image :
#    cam_pose : 6D pose of camera (in world frame)
#    original_camera : Camera intrinsics
gt_poses, ids, gt_rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
GL.view_rgb_image(gt_rgb_image;in_255=true)

IV.imshow(GL.view_rgb_image(gt_rgb_image;in_255=true))

renderer = GL.setup_renderer(original_camera, GL.RGBMode();gl_version=(3,3))
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id])
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
    GL.load_object!(renderer, mesh)
end

colors = [I.colorant"lightsalmon", I.colorant"goldenrod1",I.colorant"darkseagreen1", I.colorant"hotpink", I.colorant"cadetblue1", I.colorant"navajowhite4", I.colorant"firebrick2"]
rgb_image,depth_image = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE; colors=colors[1:length(ids)])
i = GL.view_rgb_image(rgb_image)
IV.imshow(i)

renderer = GL.setup_renderer(original_camera, GL.TextureMixedMode(); gl_version=(3,3))
texture_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id];tex_path=texture_paths[id])
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
    GL.load_object!(renderer, mesh)
end

table_bbox = S.Box(50.0, 1.0, 50.0)
table_mesh = GL.box_mesh_from_dims(T.get_dims(table_bbox))
GL.load_object!(renderer, table_mesh)

table_pose = Pose([0.0, 10.0, 100.0], IDENTITY_ORN)
rgb_image,depth_image = GL.gl_render(renderer, vcat(ids,22), vcat(gt_poses,table_pose), IDENTITY_POSE;
    colors=[I.colorant"tan" for _ in 1:(length(ids) + 1)]
)
i = GL.view_rgb_image(rgb_image)
IV.imshow(i)



# lims = (40.0, 150.0)
# d = clamp.(depth_image, lims...)
# p1 = Plots.heatmap(d; c=:thermal, clim=lims, ylim=(0, 480), xlim=(0, 640),margin = 0mm,
#     yflip=true, aspect_ratio=:equal, legend = :none, yticks=false, xticks=false, xaxis=false, yaxis=false)
# Plots.savefig("$(IDX)_rendered_depth.png")
# d = clamp.(gt_depth_image, lims...)
# @show T.min_max(d[:])
# p2 = Plots.heatmap(d; c=:thermal,clim=lims,  ylim=(0, 480), xlim=(0, 640),margin = 0mm,
#     yflip=true, aspect_ratio=:equal, legend = :none, yticks=false, xticks=false, xaxis=false, yaxis=false)
# Plots.savefig("$(IDX)_real_depth.png")

# # +
# d = clamp.(depth_image, lims...)
# d = map(x->if (x < lims[2]) x + Gen.normal(0.0, 5.0) else x end, d)

# p1 = Plots.heatmap(d; c=:thermal, clim=lims, ylim=(0, 480), xlim=(0, 640),margin = 0mm,
#     yflip=true, aspect_ratio=:equal, legend = :none, yticks=false, xticks=false, xaxis=false, yaxis=false)
# Plots.savefig("$(IDX)_rendered_depth_with_noise.png")
# p1


