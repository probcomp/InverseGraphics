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
    p = {T.floating_pose_addr(1)} ~ T.uniformPose(-0.001, 0.001, -0.001, 0.001, 4.999, 5.001)
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
constraints = Gen.choicemap(:id => 13);
args = (renderer, resolution, 0.01, 0.01);
gt_trace, _ = Gen.generate(model, args, constraints);
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
        T.geodesicHopf(StaticArrays.SVector(dir), ang)
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


xyz = [
    rotations_to_enumerate_over[i,1] * [0, 0, 1]
    for i in 1:size(rotations_to_enumerate_over)[1] 
];
log_weights_xyz = maximum(scores,dims=2)[:,1]
log_weights_xyz = log_weights_xyz .- Gen.logsumexp(log_weights_xyz);
weights_xyz = exp.(log_weights_xyz)


# PyCall.py"""
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib import cm

# def viz(points, weights):
#     fig = plt.figure()
#     ax = fig.add_subplot( 1, 1, 1, projection='3d')

#     u = np.linspace( 0, 2 * np.pi, 120)
#     v = np.linspace( 0, np.pi, 60 )

#     # create the sphere surface
#     XX = 10 * np.outer( np.cos( u ), np.sin( v ) )
#     YY = 10 * np.outer( np.sin( u ), np.sin( v ) )
#     ZZ = 10 * np.outer( np.ones( np.size( u ) ), np.cos( v ) )

#     WW = XX.copy()
#     for i in range( len( XX ) ):
#         for j in range( len( XX[0] ) ):
#             x = XX[ i, j ]
#             y = YY[ i, j ]
#             z = ZZ[ i, j ]
#             WW[ i, j ] = near(np.array( [x, y, z ] ), pointList, 3)
#     WW = WW / np.amax( WW )
#     myheatmap = WW

#     # ~ ax.scatter( *zip( *pointList ), color='#dd00dd' )
#     ax.plot_surface( XX, YY,  ZZ, cstride=1, rstride=1, facecolors=cm.jet( myheatmap ) )
#     plt.show() 

# reordered_traces = traces[sortperm(weights;rev=true)];

# i = 1;
# best_trace = reordered_traces[i][1];
# viz_trace(best_trace);
# """




import Plots