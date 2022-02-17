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
import PyCall
import PyCall: PyObject
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end

V.setup_visualizer()

function CameraIntrinsics(step_metadata::PyCall.PyObject)::GL.CameraIntrinsics
    width, height = step_metadata.camera_aspect_ratio
    aspect_ratio = width / height

    # Camera principal point is the center of the image.
    cx, cy = width / 2.0, height / 2.0

    # Vertical field of view is given.
    fov_y = deg2rad(step_metadata.camera_field_of_view)
    # Convert field of view to distance to scale by aspect ratio and
    # convert back to radians to recover the horizontal field of view.
    fov_x = 2 * atan(aspect_ratio * tan(fov_y / 2.0))

    # Use the following relation to recover the focal length:
    #   FOV = 2 * atan( (0.5 * IMAGE_PLANE_SIZE) / FOCAL_LENGTH )
    fx = cx / tan(fov_x / 2.0)
    fy = cy / tan(fov_y / 2.0)

    clipping_near, clipping_far = step_metadata.camera_clipping_planes

    GL.CameraIntrinsics(width, height,
        fx, fy, cx, cy,
        clipping_near, clipping_far)
end
get_depth_image_from_step_metadata(step_metadata::PyObject) = Matrix(last(step_metadata.depth_map_list))
get_rgb_image_from_step_metadata(step_metadata::PyObject) = Int.(numpy.array(last(step_metadata.image_list)))

mcs = PyCall.pyimport("machine_common_sense")
controller = mcs.create_controller("/Users/nishadgothoskar/GenPRAM_workspace/GenPRAM/assets/config_level1.ini")


scene_data, status = mcs.load_scene_json_file(
    "/Users/nishadgothoskar/GenPRAM_workspace/GenPRAM/assets/eval_4_validation/eval_4_validation_containers_0001_11.json")
step_metadata_list = []
step_metadata = controller.start_scene(scene_data)
push!(step_metadata_list, step_metadata)
camera = CameraIntrinsics(step_metadata)

step_metadata = controller.step("RotateRight")
push!(step_metadata_list, step_metadata)

depth_images = get_depth_image_from_step_metadata.(step_metadata_list)
obs_clouds = map(x->GL.depth_image_to_point_cloud(x,camera), depth_images);

# +
c1 = obs_clouds[1]
c2 = obs_clouds[2]
c1 = T.voxelize(c1, 0.1)
c2 = T.voxelize(c2, 0.1)

p1 = IDENTITY_POSE

pose_deltas
p2 = Pose([0.0, 0.0, 0.0], R.RotY(deg2rad(10.0)))

V.reset_visualizer()
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(T.move_points_to_frame_b(c2,p2) ./ 10.0; color=I.colorant"black", channel_name=:h2)

