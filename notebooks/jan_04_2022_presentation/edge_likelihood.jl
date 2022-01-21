import Revise
import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import LinearAlgebra as LA
import OpenCV as CV
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

rgb_image_p = permutedims(rgb_image,(3,1,2));
rgb_edges = CV.Canny(rgb_image_p, 50.0,600.0)
GL.view_depth_image(rgb_edges[1,:,:])

d = reshape(gt_depth_image, (1,size(gt_depth_image)...))
m = 200.0
d = round.(UInt8,clamp.(d,0.0, m) ./ m .* 255.0)
GL.view_depth_image(d[1,:,:])

d_edges = CV.Canny(reshape(d, (1,size(gt_depth_image)...)), 1.0,500.0)
GL.view_depth_image(d_edges[1,:,:])

GL.view_depth_image(diff(d;dims=2)[1,:,:])


GL.view_depth_image(diff(diff(d;dims=3);dims=3)[1,:,:])



# +
normalize(v) = (v./sqrt(sum(v.^2)))

c = GL.depth_image_to_point_cloud(gt_depth_image, camera; flatten=false)
h,w = size(c)[1:2]
m = 5
dot_prod = zeros(h,w)
for i in (1+m):(h-m)
    for j in (1+m):(w-m)
        v1 = normalize(c[i-m,j,:] .- c[i,j,:])
        v2 = normalize(c[i,j,:] .- c[i+m,j,:])
        dot_prod[i,j] = sum(v1 .* v2)
    end
end
dot_prod
# -

GL.view_depth_image(dot_prod .< 0.01)

c = GL.depth_image_to_point_cloud(gt_depth_image, camera; flatten=false)
h,w = size(c)[1:2]
m = 5
dot_prod = zeros(h,w)
for i in (1+m):(h-m)
    for j in (1+m):(w-m)
        v1 = normalize(c[i,j-m,:] .- c[i,j,:])
        v2 = normalize(c[i,j,:] .- c[i,j+m,:])
        dot_prod[i,j] = sum(v1 .* v2)
    end
end
dot_prod

GL.view_depth_image(dot_prod .> 0.6)

gt_cloud = T.move_points_to_frame_b(GL.depth_image_to_point_cloud(gt_depth_image, camera),cam_pose)
gt_cloud = GL.discretize(gt_cloud, 0.5)
V.viz(gt_cloud)

# +
import NearestNeighbors

pcloud = gt_cloud
k = 5
idxs, dists = NearestNeighbors.knn(NearestNeighbors.KDTree(pcloud),pcloud, k)
normals = hcat([LA.normalize( pcloud[:, idxs[i]]' \ ones(k)) for i=1:size(pcloud)[2]]...)
reshaped = reshape(normals, (3,size(gt_depth_image)...))
@show size(reshaped)
# -

mi,ma = minimum(reshaped),maximum(reshaped)
reshaped = (reshaped .- mi) ./ (ma - mi)
mi,ma = minimum(reshaped),maximum(reshaped)
@show mi, ma
img = I.colorview(I.RGB, reshaped)

c = GL.depth_image_to_point_cloud(gt_depth_image, camera; flatten=false)
h,w = size(c)[1:2]
m = 5
dot_prod = zeros(h,w)
for i in (1+m):(h-m)
    for j in (1+m):(w-m)
        v1 = normalize(c[i,j-m,:] .- c[i,j,:])
        v2 = normalize(c[i,j,:] .- c[i,j+m,:])
        dot_prod[i,j] = sum(v1 .* v2)
    end
end
dot_prod
