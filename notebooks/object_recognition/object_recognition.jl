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
for id in 1:21
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


Gen.@gen function model(model_params::T.SceneModelParameters)
    camera_pose = {T.camera_pose_addr()} ~ T.uniformPose(-1000.0,1000.0,-1000.0,1000.0,-1000.0,1000.0)
    p = {T.floating_pose_addr(1)} ~ T.uniformPose(-2.0, 2.0, -2.0, 2.0, 7.0, 13.0)
    c =  model_params.get_cloud(
        [p], model_params.ids, camera_pose, 1
    )
    obs_cloud = {T.obs_addr()} ~ T.uniform_mixture_from_template(
                                                c,
                                                0.01,
                                                resolution,
                                                (-100.0,100.0,-100.0,100.0,-100.0,300.0))
    return (rendered_cloud=c,
            obs_cloud=obs_cloud)
end

function viz_trace(trace)
    V.reset_visualizer()
    V.viz(Gen.get_retval(trace).rendered_cloud; color=I.colorant"red", channel_name=:gen);
    V.viz(Gen.get_retval(trace).obs_cloud; color=I.colorant"blue", channel_name=:obs);
end

function render_func(i,p, cam_pose)
    test_depth_image = GL.gl_render(renderer, i, p, cam_pose);
    c = GL.depth_image_to_point_cloud(test_depth_image, intrinsics);
    c = GL.voxelize(c, resolution)
end

hypers = T.Hyperparams(
    slack_dir_conc=1000.0, # Flush Contact parameter of orientation VMF
    slack_offset_var=0.5, # Variance of zero mean gaussian controlling the distance between planes in flush contact
    p_outlier=0.01, # Outlier probability in point cloud likelihood
    noise=0.1, # Spherical ball size in point cloud likelihood
    resolution=resolution, # Voxelization resolution in point cloud likelihood
    parent_face_mixture_prob=0.99, # If the bottom of the can is in contact with the table, the an object
    # in contact with the can is going to be contacting the top of the can with this probability.
    floating_position_bounds=(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0), # When an object is
    # not in contact with another object, its pose is drawn uniformly with position bounded by the given extent.
)

particles = [
    let 
        params = T.SceneModelParameters(
            boxes=[S.Box(40.0,40.0,0.1)], # Bounding box size of objects
            ids=[i],
            get_cloud=
            (poses, ids, cam_pose, i) -> render_func(ids, poses, cam_pose), # Get cloud function
            hyperparams=hypers,
            N=1 # Number of meshes to sample for pseudomarginalizing out the shape prior.
        );
        Gen.generate(model, (params,), Gen.choicemap(T.obs_addr() => c, T.camera_pose_addr() => IDENTITY_POSE))[1]
    end
    for i in 1:length(obj_paths)
];
viz_trace(particles[1]);

particles =map(t -> T.icp_move(t, 1)[1], particles);


V.reset_visualizer()





IV.imshow(GL.view_depth_image(test_depth_image))