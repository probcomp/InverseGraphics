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
import ImageView as IV


YCB_DIR = joinpath(dirname(dirname(pathof(T))),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)

camera = GL.CameraIntrinsics()
renderer = GL.setup_renderer(camera, GL.DepthMode(); gl_version=(3,3))
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR)
for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id])
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
    GL.load_object!(renderer, mesh)
end


inferred_poses = [
    inv(Pose([0.0, 0.0, 10.0], R.RotY(ang)))
    for ang in deg2rad.([0.0, 90.0, 180.0, 270.0])
]
obj_id = 2
depth_images = [
    GL.gl_render(renderer, [obj_id], [IDENTITY_POSE], cam_pose) for cam_pose in inferred_poses
];
IV.imshow(depth_images[1])
clouds = [
    GL.depth_image_to_point_cloud(d, camera) for d in depth_images
];

pose_adjusted_cloud = hcat([T.move_points_to_frame_b(c,p) for (c,p) in zip(clouds, inferred_poses)]...);
V.viz(pose_adjusted_cloud ./ 10.0)

RESOLUTION = 0.1
mins,maxs = T.min_max(pose_adjusted_cloud)
@show mins, maxs
mins,maxs = floor.(Int,mins ./ RESOLUTION) * RESOLUTION , ceil.(Int, maxs ./RESOLUTION) * RESOLUTION
mins .-= RESOLUTION * 5
maxs .+= RESOLUTION * 5

dimensions = [length(collect(mins[1]:RESOLUTION:maxs[1])),
              length(collect(mins[2]:RESOLUTION:maxs[2])),
              length(collect(mins[3]:RESOLUTION:maxs[3]))]
ALL_VOXELS = hcat([[a,b,c] for a in collect(mins[1]:RESOLUTION:maxs[1])
                           for b in collect(mins[2]:RESOLUTION:maxs[2])
                           for c in collect(mins[3]:RESOLUTION:maxs[3])]...);
@show size(ALL_VOXELS)


alphas = zeros(size(ALL_VOXELS)[2])
betas = zeros(size(ALL_VOXELS)[2])
for t in 1:length(inferred_poses)
    occ, occl, free = T.get_occ_ocl_free(T.get_points_in_frame_b(ALL_VOXELS, inferred_poses[t]), camera, depth_images[t], 2*RESOLUTION)
    alphas[occ] .+= 1.0
    betas[free] .+= 1.0
end
p = alphas ./ (alphas .+ betas)
p[isnan.(p)] .= 0.5;


V.reset_visualizer()
V.viz(ALL_VOXELS[:, p.>0.5] ./ 10.0; color=I.colorant"black", channel_name=:occupied)
V.viz(ALL_VOXELS[:, p.==0.5]./ 10.0; color=I.colorant"red", channel_name=:uncertain)

