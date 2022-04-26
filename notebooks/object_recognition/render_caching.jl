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
import Open3DVisualizer as V
import MeshCatViz as MV
import Plots
import StaticArrays
import PyCall

MV.setup_visualizer()

intrinsics = GL.CameraIntrinsics();
intrinsics = GL.scale_down_camera(intrinsics, 4);

renderer = GL.setup_renderer(intrinsics, GL.DepthMode(), gl_version=(3,3));
YCB_DIR = joinpath(pwd(),"data")
world_scaling_factor = 10.0
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR);
for id in 1:21
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id]);
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, zeros(3));
    mesh.vertices = vcat(mesh.vertices[1,:]',-mesh.vertices[3,:]',mesh.vertices[2,:]')
    GL.load_object!(renderer, mesh);
end
names = T.load_ycb_model_list(YCB_DIR)



unit_sphere_directions = T.fibonacci_sphere(300);
other_rotation_angle = collect(0:0.1:(2*π));
rotations_to_enumerate_over = [
    let
        T.geodesicHopf_select_axis(StaticArrays.SVector{3}(dir), ang, 1)
    end
    for dir in unit_sphere_directions, ang in other_rotation_angle
];
@show size(rotations_to_enumerate_over)

id = 13
images = (orn -> GL.gl_render(
        renderer, [id], [Pose([0.0, 0.0, 5.0], orn)], IDENTITY_POSE
    )).(rotations_to_enumerate_over);
clouds = [
    GL.depth_image_to_point_cloud(d,intrinsics) for d in images
];

p = T.uniformPose(-0.001, 0.001, -0.001, 0.001, 4.999, 5.001)
d = GL.gl_render(
    renderer, [id], [p], IDENTITY_POSE
)
actual_cloud = GL.depth_image_to_point_cloud(d,intrinsics) 

dirs = hcat(unit_sphere_directions...)
idx1 = argmin(sum((dirs .- (p.orientation * [1,0,0])).^2, dims=1))[2]
idx2 = argmin([abs(R.rotation_angle(inv(p.orientation) * r)) for r in rotations_to_enumerate_over[idx1,:]])
closest_orientation = rotations_to_enumerate_over[idx1,idx2]

# Show that the orientations are close
# V.open_window()
# V.add(V.make_axes())
# V.add(V.move_mesh_to_pose(V.make_axes(0.3), Pose(p.pos,closest_orientation) ))
# V.add(V.move_mesh_to_pose(V.make_axes(0.3), p ))
# V.run()


Gen.@gen function model(renderer,  resolution, p_outlier, ball_radius)
    id ~ Gen.categorical(ones(21) ./ 21)
    p = {T.floating_pose_addr(1)} ~ T.uniformPose(-0.001, 0.001, -0.001, 0.001, 4.999, 5.001)
    depth_image =  GL.gl_render(
        renderer, [id], [p], IDENTITY_POSE
    )
    cloud = GL.depth_image_to_point_cloud(depth_image, renderer.camera_intrinsics);
    voxelized_cloud = GL.voxelize(cloud, resolution)
    obs_cloud = {T.obs_addr()} ~ T.uniform_mixture_from_template(
                                                voxelized_cloud,
                                                p_outlier,
                                                ball_radius,
                                                (-100.0,100.0,-100.0,100.0,-100.0,300.0))

    return (id =id, pose=p, cloud=cloud, voxelized_cloud=voxelized_cloud, depth_image=depth_image, obs_cloud=obs_cloud)
end



resolution = 0.05
args = (renderer, resolution, 0.1, 0.3);


traces = (orn -> Gen.generate(model, args,
        Gen.choicemap(T.obs_addr() => gt_cloud,
        :id => gt_trace[:id],
        T.floating_pose_addr(1) => Pose([0.0, 0.0, 5.0], orn)))).(rotations_to_enumerate_over);
scores = (i -> i[2]).(traces);





function viz_trace(trace)
    MV.reset_visualizer()
    MV.viz(Gen.get_retval(trace).voxelized_cloud  ./ 10.0; color=I.colorant"red", channel_name=:gen);
    MV.viz(Gen.get_retval(trace).obs_cloud ./ 10.0; color=I.colorant"blue", channel_name=:obs);
end