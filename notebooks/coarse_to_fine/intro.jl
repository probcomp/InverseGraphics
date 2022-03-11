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
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end

# Initialize the renderer
V.setup_visualizer()

# Loading the YCB object models
YCB_DIR = joinpath(dirname(dirname(pwd())),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR);

# Initialize the canera intrinsics and renderer that will render using those intrinsics.
camera = GL.CameraIntrinsics()
renderer = GL.setup_renderer(camera, GL.DepthMode())
resolution = 0.05
for id in all_ids
    cloud = id_to_cloud[id]
    mesh = GL.mesh_from_voxelized_cloud(GL.voxelize(cloud, resolution), resolution)
    GL.load_object!(renderer, mesh)
end
@show camera;

object_id = 15
pose = Pose([0.0, 0.0, 5.0], R.RotXYZ(0.1, 0.2, -0.4))
camera_pose = IDENTITY_POSE
# Render object of type object_id at the specified pose viewed from a camera at camera_pose.
gt_depth_image = GL.gl_render(renderer, [object_id], [pose], camera_pose)
# Create point cloud corresponding to that rendered depth image.
c = GL.depth_image_to_point_cloud(gt_depth_image, camera)
# Visualize that point cloud.
V.viz(c)
GL.view_depth_image(gt_depth_image)

# +
# One way to do coarsen is to scale down the resolution of the rendered image

# Here we scale down the camera by a factor of 6.
scaled_camera = GL.scale_down_camera(camera, 6)

# Set the renderer to now have those scaled down intrinsics.
GL.set_intrinsics!(renderer, scaled_camera)

# And render the same image as above.
gt_depth_image = GL.gl_render(renderer, [object_id], [pose], camera_pose)

# Remember to revert the intrinsics inside of the renderer.
GL.set_intrinsics!(renderer, camera)

# Create point cloud corresponding to that rendered depth image.
c = GL.depth_image_to_point_cloud(gt_depth_image, scaled_camera)
# Visualize that point cloud.
V.viz(c)
img = GL.view_depth_image(gt_depth_image)
img = I.imresize(img, (camera.height, camera.width))

# +
# Let's see how resolution effects the rendering speed

GL.set_intrinsics!(renderer, camera)
@time gt_depth_image = GL.gl_render(renderer, [object_id], [pose], camera_pose)

# Set the renderer to now have those scaled down intrinsics.
GL.set_intrinsics!(renderer, scaled_camera)
# And render the same image as above.
@time gt_depth_image = GL.gl_render(renderer, [object_id], [pose], camera_pose)

# Remember to revert the intrinsics inside of the renderer.
GL.set_intrinsics!(renderer, camera)
# -

# Sketch of the structure of a simplified single object 3DP3 generative model.
Gen.@gen function single_object_model(resolution)
    i ~ categorical(1:21)
    p ~ T.uniformPose(-10,10,-10,10,-10,10)
    rendered_d = GL.gl_render(renderer, [i], [p], IDENTITY_POSE)
    d ~ T.uniform_mixture_from_template(
        rendered_d, 0.0001, resolution,
    (-100.0, 100.0, -100.0, 100.0,-100.0,100.0))
end


# Below, I show how this likelihood is computed between two point clouds (the latent and observed clouds).
score_clouds(obs_cloud, latent_cloud, resolution) = Gen.logpdf(
    T.uniform_mixture_from_template,
    obs_cloud, latent_cloud, 0.0001, resolution,
    (-100.0, 100.0, -100.0, 100.0,-100.0,100.0))

# Two randomly generated point clouds.
c = rand(3,10000) * 10.0;
c2 = rand(3,10000) * 10.0;

@show score_clouds(c, c2, 0.1)
@show score_clouds(c, c, 0.1)

# Lets see how likelihood changes with the size of the spheres.
@time score_clouds(c, c2, 0.1)
@time score_clouds(c, c2, 2.0)

# +
# Also to make nice visualizations, you can rendering RGB mode!
# -

renderer = GL.setup_renderer(camera, GL.RGBMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id])
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
    GL.load_object!(renderer, mesh)
end

rgb_image, d = GL.gl_render(renderer, [12], [Pose([0.0, 0.0, 5.0], R.RotXYZ(0.6, -0.2, -0.4))], IDENTITY_POSE; colors=[I.colorant"red"])
GL.view_rgb_image(rgb_image)
