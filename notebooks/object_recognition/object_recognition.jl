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
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end

V.setup_visualizer()

intrinsics = GL.CameraIntrinsics();
# intrinsics = GL.scale_down_camera(intrinsics, 4);

renderer = GL.setup_renderer(intrinsics, GL.DepthMode());
YCB_DIR = joinpath(pwd(),"data")
world_scaling_factor = 10.0
obj_paths = T.load_ycb_model_obj_file_paths(YCB_DIR);
for id in all_ids
    mesh = GL.get_mesh_data_from_obj_file(obj_paths[id]);
    mesh = T.scale_and_shift_mesh(mesh, world_scaling_factor, zeros(3));
    mesh.vertices = vcat(mesh.vertices[1,:]',-mesh.vertices[3,:]',mesh.vertices[2,:]')
    GL.load_object!(renderer, mesh);
end


resolution  = 0.05

test_depth_image = GL.gl_render(renderer, [5], [Pose([0.0, 0.0, 10.0], IDENTITY_ORN)], IDENTITY_POSE);
c = GL.depth_image_to_point_cloud(test_depth_image, intrinsics);
c = GL.voxelize(c, resolution)
V.viz(c);

Gen.@gen function model(render_func, radius)
    pose ~ T.uniformPose(-2.0, 2.0, -2.0, 2.0, 7.0, 13.0)
    id ~ Gen.categorical(ones(21)./21)
    c = render_func(id, pose);
    obs_cloud ~ T.uniform_mixture_from_template(
        c,
        0.01, 
        radius,
        (-100.0, 100.0, -100.0, 100.0, -100.0, 100.0)
    )
    (pose=pose, id=id, rendered_cloud=c, obs_cloud=obs_cloud)
end


function viz_trace(trace)
    V.reset_visualizer()
    V.viz(Gen.get_retval(trace).rendered_cloud; color=I.colorant"red", channel_name=:gen);
    V.viz(Gen.get_retval(trace).obs_cloud; color=I.colorant"blue", channel_name=:obs);
end

function render_func(i,p)
    test_depth_image = GL.gl_render(renderer, [i], [p], IDENTITY_POSE);
    c = GL.depth_image_to_point_cloud(test_depth_image, intrinsics);
    c = GL.voxelize(c, resolution)
end

particles = [
    Gen.generate(model, (render_func, resolution*2.0), Gen.choicemap(:obs_cloud => c, :id => i))[1]
    for i in 1:length(obj_paths)
];
viz_trace(particles[2]);





IV.imshow(GL.view_depth_image(test_depth_image))