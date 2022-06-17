import InverseGraphics as T
import PyCall
import Open3DVisualizer as V
import MeshCatViz as MV
import NearestNeighbors as NN
import Gen
import Statistics: mean
MV.setup_visualizer()



YCB_DIR = joinpath(dirname(dirname(pathof(T))),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)


IDX = 668
@show T.get_ycb_scene_frame_id_from_idx(YCB_DIR, IDX)
gt_poses, ids, gt_rgb_image, gt_depth_image, cam_pose, original_camera = T.load_ycbv_scene_adjusted(
    YCB_DIR, IDX, world_scaling_factor, id_to_shift
);

r = T.GL.view_rgb_image(gt_rgb_image;in_255=true)
T.FileIO.save("observed_image.png", r)


camera = T.scale_down_camera(original_camera, 4)
renderer = T.GL.setup_renderer(camera, T.GL.DepthMode())
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR);
for id in all_ids
    mesh = T.GL.get_mesh_data_from_obj_file(obj_paths[id])
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, id_to_shift[id])
    T.GL.load_object!(renderer, mesh)
end

function visualize_query_img(img)
    img_reshaped = reshape(img, (size(img)[1:2]..., 3, 4))
    img_reshaped = mean(img_reshaped;dims=4)[:,:,:,1]
    img_reshaped = img_reshaped ./ (maximum(abs.(img_reshaped)) + 1e-9)
    0.5 .+ (img_reshaped .* 0.5)
end


for obj_idx in ids
    np = PyCall.pyimport("numpy")
    data = np.load("/home/nishadg/mit/surfemb/$(obj_idx-1)_test_data.npz")
    query_img = data.get("query_img")
    verts_norm_subsampled = data.get("verts_norm_subsampled")
    obj_keys = permutedims(data.get("obj_keys"))
    scale = data.get("scale")
    offset = data.get("offset")

    vertices = (permutedims(verts_norm_subsampled)) .* scale ./ 100.0 .+ (offset ./ 100.0)

    r = T.GL.view_rgb_image(visualize_query_img(query_img))
    T.FileIO.save("$(obj_idx)_descriptor_image.png", r)


    @show names[obj_idx]

    # v1 = id_to_cloud[obj_idx]
    # vertices = (permutedims(verts_norm_subsampled)) .* scale ./ 100.0 .+ (offset ./ 100.0)
    # V.open_window()
    # V.make_point_cloud(v1; color=T.I.colorant"red")
    # V.make_point_cloud(vertices; color=T.I.colorant"blue")


    tree = NN.KDTree(vertices)


    T.GL.activate_renderer(renderer)
    idx_in_list = findfirst(ids .== obj_idx)
    gt_pose = gt_poses[idx_in_list]
    depth_image = T.GL.gl_render(renderer, [obj_idx], [gt_pose], T.IDENTITY_POSE)
    r = T.GL.view_rgb_image(gt_rgb_image;in_255=true)
    d = T.I.imresize(T.GL.view_depth_image(depth_image), size(r))
    T.FileIO.save("$(obj_idx)_gt_overlay.png",  T.mix(d,r, 0.5))


    log_denominator = zeros(size(query_img)[1:2]...);
    Threads.@threads for i in 1:size(query_img, 1) 
        Threads.@threads for j in 1:size(query_img, 2)
            log_denominator[i,j] = Gen.logsumexp(permutedims(query_img[i,j,:]) * obj_keys)
        end
    end

    query_img_flattened = permutedims(reshape(query_img, (prod(size(query_img)[1:2]), size(query_img, 3))))

    test_poses = vcat([T.gaussianVMF(gt_pose, 0.1, 200.0) for _ in 1:5000], gt_pose)
    score_func(p) = T.get_neural_descriptor_score(renderer, tree, 2, p, query_img_flattened, size(query_img), obj_keys, log_denominator, original_camera) 
    scores = [score_func(p)[1] for p in test_poses];

    best_pose = test_poses[argmax(scores)]
    depth_image = T.GL.gl_render(renderer, [obj_idx], [best_pose], T.IDENTITY_POSE)
    r = T.GL.view_rgb_image(gt_rgb_image;in_255=true)
    d = T.I.imresize(T.GL.view_depth_image(depth_image), size(r))
    T.FileIO.save("$(obj_idx)_inferred_overlay.png",  T.mix(d,r, 0.5))
    @show argmax(scores)
end

score_func(best_pose)

function visualize_query_img(img)
    img_reshaped = reshape(img, (size(img)[1:2]..., 3, 4))
    img_reshaped = mean(img_reshaped;dims=4)[:,:,:,1]
    img_reshaped = img_reshaped ./ maximum(abs.(img_reshaped))
    0.5 .+ (img_reshaped .* 0.5)
end
r = T.GL.view_rgb_image(gt_rgb_image;in_255=true)
T.FileIO.save("observed_image.png", r)

r = T.GL.view_rgb_image(visualize_query_img(query_img))
T.FileIO.save("descriptor_image.png", r)

d = T.I.imresize(T.GL.view_depth_image(depth_image), size(r))
T.FileIO.save("worst.png", T.mix(d,r, 0.5))

normalize_img(query_img)










score_func(best_pose)

worst_pose = test_poses[argmin(scores)]
@show argmin(scores)
depth_image = T.GL.gl_render(renderer, [2], [worst_pose], T.IDENTITY_POSE)
r = T.GL.view_rgb_image(gt_rgb_image;in_255=true)
d = T.I.imresize(T.GL.view_depth_image(depth_image), size(r))
T.FileIO.save("worst.png", T.mix(d,r, 0.5))











img = fill(T.I.colorant"white", (original_camera.height, original_camera.width))
for (x,y) in eachcol(model_point_pixel_coords)
    img[y,x] = T.I.colorant"blue"
end
T.FileIO.save("1.png", img)






MV.viz(cloud)


MV.viz()


verts_norm_subsampled = 
depth_image = T.I.imresize(depth_image, (original_camera.height, original_camera.width));



V.open_window()
V.make_point_cloud(cloud)


function get_cloud(p, camera_pose)
    depth_image = GL.gl_render(renderer, [2], [p], camera_pose)
    cloud = GL.depth_image_to_point_cloud(depth_image, camera)
    if isnothing(cloud)
        cloud = zeros(3,1)
    end
    cloud
end


