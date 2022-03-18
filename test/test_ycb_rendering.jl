import Revise
import GLRenderer as GL
import PoseComposition: Pose, IDENTITY_ORN, IDENTITY_POSE
import Rotations as R

import InverseGraphics as T

YCB_DIR = "/home/nishadg/mit/InverseGraphics/data"
world_scaling_factor = 100.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));


SCENE = 50
IDX = 630
gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, SCENE, IDX, world_scaling_factor, id_to_shift
);
rgb = GL.view_rgb_image(rgb_image;in_255=true)

# +
cam = T.scale_down_camera(original_camera,1)
renderer = GL.setup_renderer(cam, GL.TextureMode());
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
texture_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
for id in all_ids
    p = obj_paths[id];
    mesh = GL.get_mesh_data_from_obj_file(p)
    mesh = (
        vertices= ( mesh.vertices * world_scaling_factor) .- id_to_shift[id],
        indices=mesh.indices,
        normals=mesh.normals,
        tex_coords=mesh.tex_coords,
        tex_path=texture_paths[id]
    )
    @show size(mesh.normals)
    @show size(mesh.indices)
    @show size(mesh.vertices)
    GL.load_object!(renderer, mesh)

end
# -

import Images
colors = Images.distinguishable_colors(length(ids), Images.colorant"red")
colors = map(Images.RGBA, colors)
# @time rgb, depth_image  = GL.gl_render(renderer, ids, gt_poses,colors, IDENTITY_POSE)
@time rgb, depth_image  = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE)


GL.view_depth_image(depth_image)

GL.view_rgb_image(rgb)

GL.view_rgb_image(rgb_image;in_255=true)

cam = T.scale_down_camera(original_camera,1)
renderer = GL.setup_renderer(cam, GL.RGBMode());
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
texture_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
meshes = []
for id in all_ids
    p = obj_paths[id];
    mesh = GL.get_mesh_data_from_obj_file(p)
    mesh = (
        vertices= ( mesh.vertices * world_scaling_factor) .- id_to_shift[id],
        indices=mesh.indices,
        normals=mesh.normals,
    )
    v = mesh.vertices
    mesh = GL.mesh_from_voxelized_cloud(GL.voxelize(collect(v), 0.5), 0.5)
    push!(meshes, mesh)
    GL.load_object!(renderer, mesh)
end

import Images
colors = Images.distinguishable_colors(length(ids), Images.colorant"red")
colors = map(Images.RGBA, colors)
# @time rgb, depth_image  = GL.gl_render(renderer, ids, gt_poses,colors, IDENTITY_POSE)
@time rgb, depth_image  = GL.gl_render(renderer,ids,gt_poses, colors, IDENTITY_POSE)

GL.view_depth_image(depth_image)

GL.view_rgb_image(rgb)
