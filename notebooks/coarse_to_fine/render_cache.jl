import Revise
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import NearestNeighbors
import GenDirectionalStats as GDS
import Gen
import Open3DVisualizer as V
import MeshCatViz as MV
import Rotations as R
import StaticArrays
import Serialization
import ProgressMeter
import FileIO

# Load YCB objects
YCB_DIR = joinpath(dirname(dirname(pathof(T))),"data")
world_scaling_factor = 10.0
id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);
all_ids = sort(collect(keys(id_to_cloud)));
names = T.load_ycb_model_list(YCB_DIR)


camera_original = T.GL.CameraIntrinsics()
camera = T.scale_down_camera(camera_original, 4)

# Set up renderer with scaled down intrinsics: 160x120
renderer = T.GL.setup_renderer(camera, T.GL.DepthMode();gl_version=(4,1))
resolution = 0.05
for id in all_ids
    cloud = id_to_cloud[id]
    mesh = T.GL.mesh_from_voxelized_cloud(T.GL.voxelize(cloud, resolution), resolution)
    T.GL.load_object!(renderer, mesh)
end


unit_sphere_directions = T.fibonacci_sphere(700);
other_rotation_angle = collect(0:0.1:(2*π));
rotations_to_enumerate_over = [
    let
        T.geodesicHopf_select_axis(StaticArrays.SVector(dir...), ang, 1)
    end
    for dir in unit_sphere_directions, 
        ang in other_rotation_angle
];
@show length(rotations_to_enumerate_over)
dirs = hcat(unit_sphere_directions...)


# cloud_lookup = Array{Any}(zeros(length(all_ids),size(rotations_to_enumerate_over)...))
# ProgressMeter.@showprogress for i = 1:size(rotations_to_enumerate_over,1)
# 	for id in all_ids
# 		for j = 1:size(rotations_to_enumerate_over,2)
# 			cloud_lookup[id, i,j] = let
# 	            position = [0.0, 0.0, 10.0]
# 	            pose = Pose(position, rotations_to_enumerate_over[i,j])
# 	            d = T.GL.gl_render(renderer, [id], [pose], IDENTITY_POSE);
# 	            c = T.GL.depth_image_to_point_cloud(d, camera)
# 	            c = T.get_points_in_frame_b(c, pose)
# 	            c = T.voxelize(c, 0.005)
# 	        end
# 	    end
# 	end
# end

# Serialization.serialize("render_caching_data.data", (cloud_lookup, rotations_to_enumerate_over, unit_sphere_directions, other_rotation_angle, dirs))


cloud_lookup, rotations_to_enumerate_over, unit_sphere_directions, other_rotation_angle, dirs = Serialization.deserialize("render_caching_data.data")

function lookup_closet_cache_index(p)
	idx1 = argmin(sum((dirs .- (p.orientation * [1,0,0])).^2, dims=1))[2]
	idx2 = argmin([abs(R.rotation_angle(inv(p.orientation) * r)) for r in rotations_to_enumerate_over[idx1,:]])
	idx1,idx2
end

function get_cloud_cached(p, id)
    cam_pose = T.Pose(zeros(3),R.RotX(-asin(p.pos[2] / p.pos[3])) * R.RotY(asin(p.pos[1] / p.pos[3])))
    idx1,idx2 = lookup_closet_cache_index(inv(cam_pose) * p)
    cached_cloud = T.move_points_to_frame_b(cloud_lookup[id,idx1,idx2], p)
end

MV.setup_visualizer()

p = Pose([1.0, -1.0, 10.0], GDS.uniform_rot3())
id = 15

d = T.GL.gl_render(renderer, [id], [p], IDENTITY_POSE);
FileIO.save("1.png",T.GL.view_depth_image(d));
actual_cloud = T.voxelize(
	T.GL.depth_image_to_point_cloud(d, camera),
	0.005
);

MV.reset_visualizer()
MV.viz(actual_cloud ./ 10.0; color=T.I.colorant"green", channel_name=:a)
cached_cloud = get_cloud_cached(p, id)
MV.reset_visualizer()
MV.viz(cached_cloud ./ 10.0; color=T.I.colorant"blue", channel_name=:b)

