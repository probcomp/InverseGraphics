## import Revise
import PyCall
import GLRenderer as GL
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import LinearAlgebra
import Images as I
import GLRenderer as GL
import NearestNeighbors as NN
import Rotations as R
import DataStructures as DS
import InverseGraphics as T
import Statistics
import GenDirectionalStats as GDS
import MiniGSG as S
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end

V.setup_visualizer()

# +
camera_intrinsics = GL.CameraIntrinsics(400, 400, 1000.0, 1000.0 , 200.0, 200.0, 0.01, 30.0)
renderer = GL.setup_renderer(camera_intrinsics, GL.DepthMode())
camera_intrinsics

# box_dims = [0.2, 0.2, 0.2]
# v,n,f = GL.box_mesh_from_dims(box_dims)
# GL.load_object!(renderer, v, f)
# box_box = S.Box(box_dims...)
# nominal_feature_coords_in_obj_frame = T.get_bbox_corners(box_box)

v,n,f, = renderer.gl_instance.load_obj_parameters(
    "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2/models/025_mug/textured_simple.obj")
v *= 5.0
GL.load_object!(renderer, v, f)
nominal_feature_coords_in_obj_frame = reshape([0.32, 0.0, 0.0], (3,1))

# v,n,f, = renderer.gl_instance.load_obj_parameters(
#     "/home/nishadg/mcs/ThreeDVision.jl/data/ycbv2/models/011_banana/textured_simple.obj")
# v *= 2.0
# GL.load_object!(renderer, v, f)
# nominal_feature_coords_in_obj_frame = reshape([0.12, 0.14, -0.02], (3,1))

V.reset_visualizer()
V.viz(collect(transpose(v)))
V.viz(nominal_feature_coords_in_obj_frame;channel_name=:red,color=I.colorant"red")
# -

cube_pose = Pose(0.0, 0.0, 0.0, IDENTITY_ORN)
camera_pose = Pose(0.0,0.0,-4.0)
depth_image = GL.gl_render(renderer, [1], [cube_pose], camera_pose);
cloud = T.move_points_to_frame_b(
    GL.depth_image_to_point_cloud(depth_image, camera_intrinsics), 
    camera_pose
)

resolution = 0.015

# +
training_dataset = []

for _ in 1:1000
    cube_pose = Pose(0.0, 0.0, 0.0, GDS.uniform_rot3())
    camera_pose = Pose(0.0,0.0,-3.0)
    depth_image = GL.gl_render(renderer, [1], [cube_pose], camera_pose);
#     GL.view_depth_image(depth_image)
    
    feature_coords = T.move_points_to_frame_b(nominal_feature_coords_in_obj_frame, cube_pose)
    valid = T.are_features_visible(feature_coords, depth_image, camera_intrinsics, camera_pose)
    visible_feature_coords = feature_coords[:, valid]
#     img_copy = GL.view_depth_image(depth_image)
#     for pos in eachcol(visible_feature_coords)
#         x,y = GL.point_cloud_to_pixel_coordinates(T.get_points_in_frame_b(reshape([pos...],(3,1)), camera_pose), camera_intrinsics)
#         img_copy[y-5:y+5,x-5:x+5] .= I.colorant"red"
#     end
#     img_copy
    
    voxelized_point_cloud = T.voxelize(
        T.move_points_to_frame_b(
            GL.depth_image_to_point_cloud(depth_image, camera_intrinsics), 
            camera_pose
        )
    , resolution)
    @show size(voxelized_point_cloud)

    voxel_grid = round.(Int, voxelized_point_cloud ./ resolution)
    @show T.min_max(voxel_grid)
    
    visible_feature_grid  = round.(Int, visible_feature_coords ./ resolution)

    
    push!(training_dataset, (voxel_grid, visible_feature_grid));
end

# -

voxel_grid, visible_feature_grid = training_dataset[4]
V.reset_visualizer()
V.viz(voxel_grid * resolution)
V.viz(visible_feature_grid * resolution; channel_name=:red, color=I.colorant"red")


# +
min_coord = Int[100, 100, 100]
max_coord = Int[-100, -100, -100]

for (voxel_grid, visible_feature_grid) in training_dataset
    mi, ma = T.min_max(voxel_grid)
    min_coord = min.(min_coord, mi[:])
    max_coord = max.(max_coord, ma[:])
    if size(visible_feature_grid)[2] > 0
        mi, ma = T.min_max(visible_feature_grid)
        min_coord = min.(min_coord, mi[:])
        max_coord = max.(max_coord, ma[:])
    end
end
dimensions = (max_coord .- min_coord) .+ 1

training_dataset_2 = []
for (voxel_grid, visible_feature_grid) in training_dataset
    in_grid = zeros(Bool, dimensions...)
    for x in eachcol(voxel_grid)
        idx = (x .- min_coord .+ 1)
        in_grid[idx...] = true 
    end
    
    out_grid = zeros(Bool, dimensions...)
    for x in eachcol(visible_feature_grid)
        idx = (x .- min_coord .+ 1)
        window_half_length = 1
        l = max.((idx .- window_half_length),1)
        h = min.((idx .+ window_half_length),dimensions)
        out_grid[l[1]:h[1],l[2]:h[2],l[3]:h[3]] .= true 
    end
    
    push!(training_dataset_2, (in_grid, out_grid))
end


# for (voxel_grid, visible_feature_grid) in training_dataset

# (voxel_grid, visible_feature_grid) = training_dataset[1]

# mi, ma = T.min_max(voxel_grid)
# dimensions = (ma .- mi) .+ 1
# in_grid = zeros(Bool, dimensions...)
# for x in eachcol(voxel_grid)
#     idx = (x .- mi .+ 1)
#     in_grid[idx...] = true 
# end

# ker = 5
# padded_in_grid = fill(false,(size(in_grid) .+ ker*2)...)
# @show size(in_grid)
# @show size(padded_in_grid)
# padded_in_grid[(ker+1):end-ker,(ker+1):end-ker,(ker+1):end-ker] = in_grid

# x=[
#     padded_in_grid[i-ker:i+ker,j-ker:j+ker,k-ker:k+ker]
#     for i in (ker+1):(size(padded_in_grid)[1]-ker),
#      j in (ker+1):(size(padded_in_grid)[2]-ker),
#      k in (ker+1):(size(padded_in_grid)[3]-ker)
            
# ];
# @show size(x)
# @show Statistics.mean(map(any,x))

# +
in_grid, out_grid = training_dataset_2[2]
@show size(in_grid)
in_data = hcat([[Tuple(x)...] for x in findall(in_grid)]...)
out_data = hcat([[Tuple(x)...] for x in findall(out_grid)]...)

V.reset_visualizer()
V.viz(in_data * resolution  / 2.0)
V.viz(out_data * resolution / 2.0; channel_name=:red, color=I.colorant"red")
# -

for i in 1:100
   in_grid, out_grid = training_dataset_2[i]
    @show size(in_grid)
    in_data = hcat([[Tuple(x)...] for x in findall(in_grid)]...)
    out_data = hcat([[Tuple(x)...] for x in findall(out_grid)]...)

    V.reset_visualizer()
    V.viz(in_data * resolution  / 2.0)
    @show size(out_data)
    if size(out_data)[1] > 0
#         V.viz(out_data * resolution / 2.0; channel_name=:red, color=I.colorant"red")
    end
    sleep(0.5)
end

sum(in_grid) / length(in_grid)

PyCall.py"""
import pickle
def serialize(filename, data):
    outfile = open(filename,'wb')
    pickle.dump(data,outfile)
    outfile.close()

def deserialize(filename):
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    return new_dict
"""

PyCall.py"serialize"("data.pkl", training_dataset_2)

recovered_data = PyCall.py"deserialize"("data.pkl");

# # Visualize PyTorch Results

in_grid, out_grid = PyCall.py"deserialize"("result.pkl");

@show size(in_grid)
@show size(out_grid)

V.reset_visualizer()
i = 15
thresh = 0.7
in_data = hcat([[Tuple(x)...] for x in findall(in_grid[i,1,:,:,:] .> 0.0)]...)
out_data = hcat([[Tuple(x)...] for x in findall(out_grid[i,:,:,:,1] .> thresh )  ]...)
@show size(out_data)
V.viz(in_data * resolution ./ 3.0)
V.viz(out_data * resolution ./ 3.0; channel_name=:red, color=I.colorant"red")

for i in 1:50
    V.reset_visualizer()
    thresh = 0.7
    in_data = hcat([[Tuple(x)...] for x in findall(in_grid[i,1,:,:,:] .> 0.0)]...)
    out_data = hcat([[Tuple(x)...] for x in findall(out_grid[i,:,:,:,1] .> thresh )  ]...)
    V.viz(in_data * resolution ./ 3.0)
    if size(out_data)[1] > 0
        V.viz(out_data * resolution ./ 3.0; channel_name=:red, color=I.colorant"red")
    end
end


