import InverseGraphics as T
import Open3DVisualizer as V

intrinsics = T.GL.CameraIntrinsics()
renderer = T.GL.setup_renderer(intrinsics, T.GL.DepthMode();gl_version=(4,1))
box = T.GL.box_mesh_from_dims([1.0,2.0,3.0])
T.GL.load_object!(renderer, box)

T.GL.activate_renderer(renderer)
depth_image = T.GL.gl_render(renderer, [1],[T.Pose(0.0, 0.0, 10.0)], T.IDENTITY_POSE);
T.FileIO.save("depth.png", T.GL.view_depth_image(depth_image))

V.open_window()
V.clear()
V.make_point_cloud(T.GL.depth_image_to_point_cloud(depth_image, intrinsics))
V.make_point_cloud(rand(3,100))
V.make_axes(1.0)
V.sync()

V.destroy()
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

obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)

meshes = [
    V.make_mesh(nothing, obj_paths[id]) for id in all_ids
];

V.open_window(original_camera, cam_pose)
V.clear()
for (id, p) in zip(ids, gt_poses)
    V.add(V.move_mesh_to_pose(meshes[id], cam_pose * p))
end
V.set_camera_intrinsics_and_pose(original_camera, cam_pose)
V.sync()

img = V.capture_image()
IV.imshow(GL.view_rgb_image(img))

V.run()



global o3d = PyCall.pyimport("open3d");

V.open_window()
c = V.make_point_cloud(nothing, rand(3,1000) * 10.0; color=I.colorant"red")
V.update_geometry(c)
c = V.make_point_cloud(c, rand(3,100) * 10.0; color=I.colorant"black")
V.update_geometry(c)
img = V.capture_image()

c = V.make_point_cloud(nothing, rand(3,100) * 10.0; color=I.colorant"green")
V.add_geometry(c)
V.remove_geometry(c)

b = V.make_bounding_box(nothing, S.Box(1.0, 2.0, 3.0), IDENTITY_POSE)
V.update_geometry(b)
b = V.make_bounding_box(b, S.Box(1.0, 2.0, 3.0), Pose(zeros(3), R.RotX(0.2)))
V.update_geometry(b)
V.clear()
V.destroy()

V.open_window()
box = GL.box_mesh_from_dims([1.0, 2.0, 3.0])
m = V.make_mesh(nothing, box; color=I.colorant"blue")
c = m.transform(V.pose_to_transformation_matrix(Pose([0.1, 0.0, 0.0], R.RotX(0.1))))
V.update_geometry(c)
V.run()
V.destroy()

YCB_DIR = joinpath(dirname(dirname(pathof(T))),"data")
world_scaling_factor = 1.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)
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
obj_files = T.load_ycb_model_obj_file_paths(YCB_DIR);
meshes = [V.make_mesh(nothing, p) for p in obj_files];

adjusted_meshes = [o3d.geometry.TriangleMesh(meshes[i]).transform(
    V.pose_to_transformation_matrix(p))
for (i,p) in zip(ids, gt_poses)]

intrinsics = GL.CameraIntrinsics(
    640, 480, 1000.0, 1000.0, 320.0 - 0.5, 240.0 - 0.5, 0.01, 100.0
)
V.open_window(intrinsics, Pose([0.0, 0.0, 2.0], IDENTITY_ORN));
for m in adjusted_meshes
    V.add_geometry(m)
end
V.destroy()

V.open_window(original_camera, IDENTITY_POSE);
for m in adjusted_meshes
    V.add_geometry(m)
end
destroy()

V.open_window2(intrinsics, IDENTITY_POSE);


clear()

intrinsics = GL.CameraIntrinsics()
V.open_window(intrinsics, Pose(ones(3), R.RotX(-1.0)))

for (i,p) in zip(ids, gt_poses)
    V.add_geometry(meshes[i].transform(pose_to_transformation_matrix(cam_pose * p)))
end
clear()
