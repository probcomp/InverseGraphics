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

V.setup_visualizer()

YCB_DIR = joinpath(dirname(dirname(pwd())),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)



# +
SCENE = 58
# Load scene data.
#    gt_poses : Ground truth 6D poses of objects (in the camera frame)
#    ids      : object ids (order corresponds to the gt_poses list)
#    rgb_image, gt_depth_image :
#    cam_pose : 6D pose of camera (in world frame)
#    original_camera : Camera intrinsics
gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, SCENE,1, world_scaling_factor, id_to_shift
);
gt_depth_images = [
    let
        gt_poses, ids, rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
            YCB_DIR, SCENE,i, world_scaling_factor, id_to_shift
        ); 
        gt_depth_image
    end
    for i in 2:100
];
obs_clouds = [
    let
        obs_cloud = GL.depth_image_to_point_cloud(gt_depth_image, original_camera)
        obs_cloud = T.voxelize(obs_cloud, 0.05)
        obs_cloud = obs_cloud[:, obs_cloud[3,:] .< 20.0]
    end
    for gt_depth_image in gt_depth_images
]

GL.view_rgb_image(rgb_image;in_255=true)

# +
obs_cloud1 = obs_clouds[1]
best_eq = T.find_table_plane(obs_cloud1)
camera_pose1 = T.camera_pose_from_table_eq(best_eq)

c1 = T.move_points_to_frame_b(obs_cloud1, camera_pose1)
V.reset_visualizer()
V.viz(c1 ./ 10.0)
# -

obs_cloud2 = obs_clouds[99]
best_eq = T.find_table_plane(obs_cloud2)
camera_pose2 = T.camera_pose_from_table_eq(best_eq)
c2 = T.move_points_to_frame_b(obs_cloud2, camera_pose2)
V.reset_visualizer()
V.viz(c2 ./ 10.0)

V.reset_visualizer()
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(c2 ./ 10.0; color=I.colorant"black", channel_name=:h2)

p = T.icp(c1, c2; iterations=30, c1_tree=nothing)

V.reset_visualizer()
V.viz(c1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(T.move_points_to_frame_b(c2,p) ./ 10.0; color=I.colorant"black", channel_name=:h2)

V.reset_visualizer()
V.viz(obs_cloud1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(obs_cloud2 ./ 10.0; color=I.colorant"black", channel_name=:h2)

dp = T.inverse_pose(camera_pose1) * p * camera_pose2
V.reset_visualizer()
V.viz(obs_cloud1 ./ 10.0; color=I.colorant"red", channel_name=:h1)
V.viz(T.move_points_to_frame_b(obs_cloud2 ,dp) ./ 10.0; color=I.colorant"black", channel_name=:h2)
