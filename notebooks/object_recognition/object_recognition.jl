import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import NearestNeighbors
import LightGraphs as LG
import ImageView as IV
import Gen
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end

V.setup_visualizer()

model_dir = "/home/nishadg/Downloads/shape_models/tiago_shapes_3d_models"
object_names = ["cube","cuboid","cylinder_small","sphere","torus","triangular_prism",]

intrinsics = GL.CameraIntrinsics()

renderer = GL.setup_renderer(intrinsics, GL.DepthMode())
for name in object_names
    mesh = GL.get_mesh_data_from_obj_file(joinpath(model_dir, "$(name).obj"))
    mesh.vertices = vcat(mesh.vertices[1,:]',mesh.vertices[3,:]',mesh.vertices[2,:]')
    GL.load_object!(renderer, mesh)
end

test_depth_image = GL.gl_render(renderer, [1], [Pose([0.0, 0.0, 70.0], IDENTITY_ORN)], IDENTITY_POSE);
V.viz(GL.depth_image_to_point_cloud(test_depth_image, intrinsics));

IV.imshow(GL.view_depth_image(test_depth_image))