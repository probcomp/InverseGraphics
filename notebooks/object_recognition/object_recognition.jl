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
    p = {T.floating_pose_addr(1)} ~ T.uniformPose(-0.001, 0.001, -0.001, 0.001, 5.001, 5.002)
    depth_image =  GL.gl_render(
        renderer, [id], [p], IDENTITY_POSE
    )
    c = GL.depth_image_to_point_cloud(depth_image, renderer.camera_intrinsics);
    c = GL.voxelize(c, resolution)
    obs_cloud = {T.obs_addr()} ~ T.uniform_mixture_from_template(
                                                c,
                                                p_outlier,
                                                ball_radius,
                                                (-100.0,100.0,-100.0,100.0,-100.0,300.0))
    return (id =id, pose=p, depth_image=depth_image, rendered_cloud=c, obs_cloud=obs_cloud)
end

function viz_trace(trace)
    MV.reset_visualizer()
    MV.viz(Gen.get_retval(trace).rendered_cloud  ./ 10.0; color=I.colorant"red", channel_name=:gen);
    MV.viz(Gen.get_retval(trace).obs_cloud ./ 10.0; color=I.colorant"blue", channel_name=:obs);
end

resolution = 0.1
args = (renderer, resolution, 0.01, resolution * 2);
gt_trace, _ = Gen.generate(model, args);
@show gt_trace[:id]
gt_cloud = Gen.get_retval(gt_trace).rendered_cloud
IV.imshow(GL.view_depth_image(Gen.get_retval(gt_trace).depth_image))
MV.reset_visualizer()
MV.viz(gt_cloud  ./ 10.0; color=I.colorant"red", channel_name=:gen);

observations = Gen.choicemap(T.obs_addr() => gt_cloud);

particles = [
    Gen.generate(model, args, observations)
    for _ in 1:5000
];

log_weights = [i[2] for i in particles];
log_weights = log_weights .- Gen.logsumexp(log_weights);
weights = exp.(log_weights)
reordered_particles = particles[sortperm(weights;rev=true)];
best_trace = reordered_particles[1][1];
@show weights[sortperm(weights,rev=true)]
@show best_trace[:id], gt_trace[:id]
viz_trace(best_trace);

i = 2
viz_trace(reordered_particles[i][1]);