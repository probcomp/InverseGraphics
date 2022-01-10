import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
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

IDX =1800
@show T.get_ycb_scene_frame_id_from_idx(YCB_DIR,IDX)
gt_poses, ids, rgb_image, gt_depth_image, cam_pose, camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, 55,1000, world_scaling_factor, id_to_shift
);
real_rgb= GL.view_rgb_image(rgb_image;in_255=true)

renderer = GL.setup_renderer(camera, GL.TextureMode())
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

rgb_data, _ = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE)
rendered_rgb = GL.view_rgb_image(rgb_data)

renderer.gl_instance.lightpos = [0, 0, 0]
rgb_data, _ = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE)
rendered_rgb = GL.view_rgb_image(rgb_data)

display(hcat(real_rgb, rendered_rgb))

# +
nominal_color_set = [
    I.colorant"red",
    I.colorant"white",
    I.colorant"gray",
    I.colorant"black",
    I.colorant"green",
    I.colorant"blue",
    I.colorant"yellow",
    I.colorant"firebrick",
    I.colorant"khaki1",
    I.colorant"tan",
    I.colorant"gold",
    I.colorant"goldenrod4",
    
]
function round_color(in_color)
    in_color
    dists = I.colordiff.(nominal_color_set, in_color)
    nominal_color_set[argmin(dists)]
end
display(nominal_color_set)
rounded_rendered_rgb = round_color.(rendered_rgb)
rounded_real_rgb = round_color.(real_rgb)
display(hcat(rounded_real_rgb, rounded_rendered_rgb))
# -


