import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import NearestNeighbors
import LightGraphs as LG
import GenDirectionalStats as GDS
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
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR);

IDX = 1800
# Load scene data.
#    gt_poses : Ground truth 6D poses of objects (in the camera frame)
#    ids      : object ids (order corresponds to the gt_poses list)
#    rgb_image, gt_depth_image :
#    cam_pose : 6D pose of camera (in world frame)
#    original_camera : Camera intrinsics
gt_poses, ids, gt_rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);
gt_rgb = GL.view_rgb_image(gt_rgb_image;in_255=true)

renderer = GL.setup_renderer(original_camera, GL.RGBBasicMode())
for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id])
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
    GL.load_object!(renderer, mesh)
end

c = I.distinguishable_colors(10)
rgb,depth = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE; colors=c)
r = GL.view_rgb_image(rgb)
i = 2
r = I.convert.(I.RGB, r)
mask = I.colordiff.(r, c[i]) .< 0.1
inv_mask = .!(mask)
@show sum(inv_mask)
img = copy(gt_rgb)
img[inv_mask] .*= 0.6
img

GL.view_depth_image(depth)

renderer = GL.setup_renderer(original_camera, GL.TextureMode())
tex_paths = T.load_ycb_model_texture_file_paths(YCB_DIR)
for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id]; tex_path=tex_paths[id])
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
    GL.load_object!(renderer, mesh)
end

rgb,depth = GL.gl_render(renderer, ids, gt_poses, IDENTITY_POSE; colors=c)
GL.view_rgb_image(rgb)

all_ids

poses = [Pose([x, y, 20.0], R.RotXYZ(pi,0,0)) for x in -5:2:5, y in -3:2:3][:]
rgb,depth = GL.gl_render(renderer, collect(1:length(all_ids)), poses[1:21], IDENTITY_POSE)
GL.view_rgb_image(rgb)

for i in 1:300
    for j in 1:300
        @show r[i,j]
        @show I.colordiff(r[i,j], c[3])
    end
end

sum((x -> x.alpha < 0.1).(r))

map(x -> I.colordiff(x,c[3]), r)

c[4]

I.colordiff(r[200,400],c[2]) 

# +
# box_dims = [1.0,1.0,1.0]
# box_box = S.Box(box_dims...)
# box = GL.box_mesh_from_dims(box_dims)

w = 500
camera = GL.CameraIntrinsics(w, w, w, w, w/2.0,w/2.0,0.001, 100.0)
@show camera
renderer = GL.setup_renderer(camera, GL.DepthMode())

for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id])
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
    GL.load_object!(renderer, mesh)
end

# -

d[d .< 3.0]

# +
pose = Pose([0.0, 0.0,Gen.uniform(3.0,5.0)], GDS.uniform_rot3())

camera_pose = IDENTITY_POSE
d = GL.gl_render(renderer, [14], [pose], camera_pose);
c = GL.depth_image_to_point_cloud(d,camera);
@show size(c)
V.reset_visualizer()
V.viz(c;color=I.colorant"black", channel_name=:h1)

ds = clamp.(d, 0, maximum(d[d .< maximum(d) - 0.1]) + 0.4)
c = GL.depth_image_to_point_cloud(d,camera);
@show size(c)
img = I.colorview(I.Gray, ds ./ maximum(ds))


# +
poses = [
    Pose([Gen.uniform(-2.0,2.0),Gen.uniform(-2.0,2.0),Gen.uniform(3.0,5.0)], GDS.uniform_rot3())
    for _ in 1:5
]

camera_pose = IDENTITY_POSE
d = GL.gl_render(renderer, [1 for _ in 1:length(poses)], poses, camera_pose);
c = GL.depth_image_to_point_cloud(d,camera);
@show size(c)
V.reset_visualizer()
V.viz(c;color=I.colorant"black", channel_name=:h1)
GL.view_depth_image(d)
# -

all_visible_features = [
    let
        feature_coords = hcat([
            T.move_points_to_frame_b(feat, p)
            for p in poses
        ]...)
        valid = T.are_features_visible(feature_coords, d, camera, camera_pose)
        visible_feature_coords = feature_coords[:, valid]
    end
    for feat in all_features
]
map(size, all_visible_features)


pixel_coords = GL.point_cloud_to_pixel_coordinates(hcat(all_visible_features...), camera)
d_copy = GL.view_depth_image(d)
for (x,y) in eachcol(pixel_coords)
   d_copy[y-3:y+3,x-3:x+3] .= I.colorant"red"
end
d_copy

# feature type idx, detection idx
function evaluate_ransac_candidate(ransac_candidate)
    obs_3d_coords = hcat([all_visible_features[i][:,j] for (i,j) in ransac_candidate]...)
    model_3d_coords = hcat([all_features[i] for (i,_) in ransac_candidate]...)
    t = T.get_transform_between_two_registered_clouds(obs_3d_coords, model_3d_coords)
    t = inv(t)
    error = sum((obs_3d_coords .- T.move_points_to_frame_b(model_3d_coords, t)).^2)
    error, t
end

function grow_feature_group(seed, all_visible_features, iterations)
    seed = copy(seed)
    for _ in 1:iterations
        for i in 1:length(all_visible_features)
            used_features = Set(map(x->x[1], seed))
            if i in used_features
                continue
            end
            found = false
            for j in 1:size(all_visible_features[i])[2]
                error,t = evaluate_ransac_candidate(vcat(seed, (i,j)))
                if error < 0.005
                   found = true
                    push!(seed, (i,j))
                    break
                end

            end
            if found
                break
            end
        end
    end
    seed
end

map(x->size(x)[2], all_visible_features)

candidate_iterative = grow_feature_group([(2,2)], all_visible_features, 4)
@show candidate_iterative
error, t = evaluate_ransac_candidate(candidate_iterative)
@show error
obs_3d_coords = hcat([all_visible_features[i][:,j] for (i,j) in candidate_iterative]...)
pixel_coords = GL.point_cloud_to_pixel_coordinates(obs_3d_coords, camera)
d_copy = GL.view_depth_image(d)
for (x,y) in eachcol(pixel_coords)
   d_copy[y-4:y+4,x-4:x+4] .= I.colorant"red"
end
d_copy


