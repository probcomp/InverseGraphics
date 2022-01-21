# -*- coding: utf-8 -*-
import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN, interp
import InverseGraphics as T
import NearestNeighbors
import Gen
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end

V.setup_visualizer()

YCB_DIR = "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2"
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));

SCENE = 55
IDX = 1000

gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, SCENE, IDX, world_scaling_factor, id_to_shift
);
rgb = GL.view_rgb_image(rgb_image;in_255=true)

import Plots
depth_obs_plot = Plots.heatmap(gt_depth_image; c=:thermal,clim=(20.0, 140.0), ylim=(0, 480), xlim=(0, 640),
    yflip=true, aspect_ratio=:equal, legend = :none, yticks=false, xticks=false, xaxis=false, yaxis=false)

# +
renderer = GL.setup_renderer(original_camera, GL.SegmentationMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)

for id in all_ids
    v,n,f,t = renderer.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    
    GL.load_object!(renderer, v, n, f
    )
end

# +
idx = 4
# id = ids[1]
p = gt_poses[1]
colors = map(I.RGBA,I.distinguishable_colors(length(ids)))

_, seg_mask = GL.gl_render(renderer, ids,gt_poses, IDENTITY_POSE)
overlay_img = fill(I.colorant"white", size(rgb))
overlay_img[seg_mask .== idx] .= rgb[seg_mask .== idx]
alpha = 0.2
T.mix(rgb, overlay_img, alpha)
# -

renderer = GL.setup_renderer(original_camera, GL.TextureMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
texture_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
for id in all_ids
    v,n,f,t = renderer.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    
    GL.load_object!(renderer, v, n, f, t,
        texture_paths[id]
    )
end


idx= 5
rgb_data, _ = GL.gl_render(renderer, [ids[idx]], [gt_poses[idx]], IDENTITY_POSE)
GL.view_rgb_image(rgb_data)

renderer = GL.setup_renderer(original_camera, GL.RGBMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
texture_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
for id in all_ids
    v,n,f = renderer.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    
    GL.load_object!(renderer, v, n, f
    )
end

start_pose = gt_poses[idx]
mod = Pose(zeros(3),inv(start_pose.orientation) * R.RotX(pi))
end_pose = start_pose * mod
e = start_pose * interp(mod, 1.0)
renderer.gl_instance.lightpos = [0.0, 0.0, -50.0]
images =[
    let
    rgb_data, _ = GL.gl_render(renderer, [ids[idx]], [e], [I.colorant"red"],IDENTITY_POSE)
    GL.view_rgb_image(rgb_data)
    end
    for e in [start_pose * interp(mod, t) for t in 0.0:0.01:1.0]
];

import FileIO
FileIO.save("first.png", images[1])
FileIO.save("last.png", images[end])
FileIO.save("t.gif",cat([images[1] for _ in 1:100]..., images..., [images[end] for _ in 1:100]..., dims=3);fps=50)

renderer = GL.setup_renderer(original_camera, GL.TextureMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
texture_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
for id in all_ids
    v,n,f,t = renderer.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    
    GL.load_object!(renderer, v, n, f, t,
        texture_paths[id]
    )
end

rgb_data, _ = GL.gl_render(renderer, [ids[idx]], [end_pose], IDENTITY_POSE)
GL.view_rgb_image(rgb_data)

rgb_data, _ = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE)
GL.view_rgb_image(rgb_data)

renderer = GL.setup_renderer(original_camera, GL.SegmentationMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
id_to_vnf = Dict()
for id in all_ids
    v,n,f,t = renderer.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    id_to_vnf[id] = (v,n,f)
    GL.load_object!(renderer, v, n, f
    )
end

# +
idx = 3
# id = ids[1]
p = gt_poses[1]
colors = map(I.RGBA,I.distinguishable_colors(length(ids)))

_, seg_mask = GL.gl_render(renderer, ids,gt_poses, IDENTITY_POSE)
overlay_img = fill(I.colorant"white", size(rgb))
overlay_img[seg_mask .== idx] .= rgb[seg_mask .== idx]
alpha = 0.2
T.mix(rgb, overlay_img, alpha)
# -

rgb = GL.view_rgb_image(rgb_image; in_255=true)
cloud = GL.depth_image_to_point_cloud(gt_depth_image, original_camera;flatten=false)[seg_mask .== idx, :]
colors = rgb[seg_mask .== idx]
V.reset_visualizer()
V.viz_colored(T.move_points_to_frame_b(collect(transpose(cloud)),cam_pose), colors)

v,n,f = id_to_vnf[ids[idx]]
# V.reset_visualizer()
V.viz_mesh(v,f,T.get_c_relative_to_a(cam_pose, gt_poses[idx]), :mesh;color=I.colorant"yellow")

v,n,f = id_to_vnf[ids[idx]]
v,n,f = GL.mesh_from_voxelized_cloud(GL.voxelize(collect(transpose(v)), 1.0), 1.0)
V.viz_mesh(v,f,T.get_c_relative_to_a(cam_pose, gt_poses[idx]), :mesh;color=I.colorant"yellow")

pose,box = T.axis_aligned_bbox_from_point_cloud(collect(transpose(v)))
point_list = T.get_bbox_segments_point_list(box, T.get_c_relative_to_a(cam_pose, gt_poses[idx]) * pose )
V.viz_box(point_list, :box)

# +
p = T.get_c_relative_to_a(cam_pose, gt_poses[idx]) * pose 
new_p = Pose([p.pos[1],p.pos[2],0.0], p.orientation)
point_list = T.get_bbox_segments_point_list(S.Box(50.0, 50.0, 0.1), new_p)

V.viz_box(point_list, :box2; color=I.colorant"black")
# -



renderer = GL.setup_renderer(original_camera, GL.SegmentationMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
id_to_vnf = Dict()
for id in all_ids
    v,n,f,t = renderer.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    id_to_vnf[id] = (v,n,f)
    GL.load_object!(renderer, v, n, f)
end

# +
idxs = 1:length(ids)
# id = ids[1]
p = gt_poses[1]
colors = map(I.RGBA,I.distinguishable_colors(length(ids)))

_, seg_mask = GL.gl_render(renderer, ids,gt_poses, IDENTITY_POSE)
overlay_img = fill(I.colorant"white", size(rgb))
overlay_img[map(x->x∈Set(idxs),seg_mask)] .= rgb[map(x->x∈Set(idxs),seg_mask)]
alpha = 0.2
T.mix(rgb, overlay_img, alpha)
# -

rgb = GL.view_rgb_image(rgb_image; in_255=true)
cloud = GL.depth_image_to_point_cloud(gt_depth_image, original_camera;flatten=false)[map(x->x∈Set(idxs),seg_mask), :]
colors = rgb[map(x->x∈Set(idxs),seg_mask)]
V.reset_visualizer()
V.viz_colored(T.move_points_to_frame_b(collect(transpose(cloud)),cam_pose), colors)

# +
p = T.get_c_relative_to_a(cam_pose, gt_poses[idx]) * pose 
new_p = Pose([0.0,0.0,0.0], p.orientation)
point_list = T.get_bbox_segments_point_list(S.Box(60.0, 60.0, 0.1), new_p)

V.viz_box(point_list, :box10; color=I.colorant"black")
# -

for idx in 1:length(ids)
    v,n,f = id_to_vnf[ids[idx]]
    pose,box = T.axis_aligned_bbox_from_point_cloud(collect(transpose(v)))
    point_list = T.get_bbox_segments_point_list(box, T.get_c_relative_to_a(cam_pose, gt_poses[idx]) * pose )
    V.viz_box(point_list, Symbol("box$(idx)"))
end

V.viz_box(point_list, :box4; color=I.colorant"black")

renderer = GL.setup_renderer(original_camera, GL.TextureMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
texture_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
for id in all_ids
    v,n,f,t = renderer.gl_instance.load_obj_parameters(
        obj_paths[id]
    )
    v = v * world_scaling_factor
    v .-= id_to_shift[id]'
    
    GL.load_object!(renderer, v, n, f, t,
        texture_paths[id]
    )
end

idx= 4
rgb_data, _ = GL.gl_render(renderer, [ids[idx]], [gt_poses[idx]], IDENTITY_POSE)
GL.view_rgb_image(rgb_data)

import GenDirectionalStats as GDS
rgb_data, _ = GL.gl_render(renderer, [ids[2]], [Pose([0.0, 0.0, 50.0], GDS.uniform_rot3())], IDENTITY_POSE)
GL.view_rgb_image(rgb_data)


