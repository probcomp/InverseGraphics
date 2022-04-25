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

function viz_trace(trace)
    MV.reset_visualizer()
    MV.viz(Gen.get_retval(trace).voxelized_cloud  ./ 10.0; color=I.colorant"red", channel_name=:gen);
    MV.viz(Gen.get_retval(trace).obs_cloud ./ 10.0; color=I.colorant"blue", channel_name=:obs);
end

resolution = 0.05
constraints = Gen.choicemap(:id => 13);
args = (renderer, resolution, 0.1, 0.3);
gt_trace, _ = Gen.generate(model, args, constraints);
@show gt_trace[:id]

gt_cloud = Gen.get_retval(gt_trace).voxelized_cloud
MV.reset_visualizer()
MV.viz(gt_cloud  ./ 10.0; color=I.colorant"red", channel_name=:gen);

observations = Gen.choicemap(T.obs_addr() => gt_cloud);

function fibonacci_sphere(samples)
    points = []
    phi = π * (3. - sqrt(5.))
 
    for i in 0:(samples-1)
        y = 1 - (i / Float64(samples - 1)) * 2
        @show y
        radius = sqrt(1 - y * y)

        theta = phi * i

        x = cos(theta) * radius
        z = sin(theta) * radius

        push!(points, (x, y, z))
    end
    return points
end

unit_sphere_directions = fibonacci_sphere(200);
other_rotation_angle = collect(0:0.2:(2*π));

rotations_to_enumerate_over = [
    let
        T.geodesicHopf_select_axis(StaticArrays.SVector(dir), ang, 1)
    end
    for dir in unit_sphere_directions, 
        ang in other_rotation_angle
];

traces = (orn -> Gen.generate(model, args,
        Gen.choicemap(T.obs_addr() => gt_cloud,
        :id => gt_trace[:id],
        T.floating_pose_addr(1) => Pose([0.0, 0.0, 5.0], orn)))).(rotations_to_enumerate_over);
scores = (i -> i[2]).(traces);

log_weights = [i[2] for i in traces];
log_weights = log_weights .- Gen.logsumexp(log_weights);
weights = exp.(log_weights)
best_trace, _ = traces[argmax(weights)];
viz_trace(best_trace);
sort(weights[:])


xyz = [
    rotations_to_enumerate_over[i,1] * [1, 0, 0]
    for i in 1:size(rotations_to_enumerate_over)[1] 
];
log_weights_xyz = maximum(scores,dims=2)[:,1]
log_weights_xyz = log_weights_xyz .- Gen.logsumexp(log_weights_xyz);
weights_xyz = exp.(log_weights_xyz)
order = sortperm(weights_xyz,rev=true)
weights_xyz = weights_xyz[order]
xyz = xyz[order]

PyCall.py"""
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
def run_viz(x,y,z,c):
    fig = plt.figure(figsize=(9, 6))
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=c)
    ax.set_title("3D scatterplot", pad=25, size=15)
    ax.set_xlabel("X") 
    ax.set_ylabel("Y") 
    ax.set_zlabel("Z")
    plt.show()
"""

PyCall.py"run_viz"( (i->i[1]).(xyz), (i->i[2]).(xyz), (i->i[3]).(xyz), weights_xyz)