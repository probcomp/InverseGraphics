import Pkg
ENV["PYTHON"] = "/usr/bin/python3"
Pkg.build("PyCall")

import GLRenderer as GL
import Images as I
import MiniGSG as S
import Rotations as R
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import InverseGraphics as T
import Gen
try
    import MeshCatViz as V
catch
    import MeshCatViz as V
end
using CoordinateTransformations
import GeometryBasics
import Colors
import MetaGraphs as MG
import GeometryBasics: Point, Mesh, GLTriangleFace, HyperSphere
import GenDirectionalStats as GDS

# YCB_DIR = joinpath(dirname(pwd()),"data")
# world_scaling_factor = 10.0
# id_to_cloud, id_to_shift, id_to_box  = T.load_ycbv_models_adjusted(YCB_DIR, world_scaling_factor);


camera = GL.CameraIntrinsics()
camera = GL.scale_down_camera(camera, 4)
renderer = GL.setup_renderer(camera, GL.DepthMode();gl_version=(3,3))

object_boxes = [S.Box(1.0, 1.0, 1.0), S.Box(1.5, 1.5, 1.5)]
for box in object_boxes
    mesh = GL.box_mesh_from_dims(T.get_dims(box))
    GL.load_object!(renderer, mesh)
end

function get_cloud(poses, ids, camera_pose)
    depth_image = GL.gl_render(renderer, ids, poses, camera_pose)
    cloud = GL.depth_image_to_point_cloud(depth_image, camera)
    if isnothing(cloud)
        cloud = zeros(3,1)
    end
    cloud
end

# Model parameters
hypers = T.Hyperparams(
    slack_dir_conc=800.0,          # Flush Contact parameter of orientation VMF
    slack_offset_var=0.2,          # Variance of zero mean gaussian controlling the distance between planes in flush contact
    p_outlier=0.01,                # Outlier probability in point cloud likelihood
    noise=0.1,                     # Spherical ball size in point cloud likelihood
    resolution=0.06,               # Voxelization resolution in point cloud likelihood
    parent_face_mixture_prob=0.99, # If the bottom of the can is in contact with the table, the an object
    # in contact with the can is going to be contacting the top of the can with this probability.
    floating_position_bounds=(-1000.0, 1000.0, -1000.0,1000.0,-1000.0,1000.0), # When an object is
    # not in contact with another object, its pose is drawn uniformly with position bounded by the given extent.
)

ids = [1, 2]

params = T.SceneModelParameters(
    # Bounding box size of objects
    boxes=object_boxes,
    # Get cloud function
    get_cloud=(poses, ids, cam_pose, i) -> get_cloud(poses, ids, cam_pose),
    hyperparams=hypers,
    # Number of meshes to sample for pseudomarginalizing out the shape prior.
    N=1,
    ids=ids,
);


constraints = Gen.choicemap()
g = T.graph_with_edges(2, [1=>2])
constraints[T.structure_addr()] = g
constraints[T.floating_pose_addr(1)] = IDENTITY_POSE
constraints[:p_outlier] = hypers.p_outlier
constraints[:noise] = hypers.noise
constraints[T.contact_addr(2, :child_face)] = :back
constraints[T.contact_addr(2, :parent_face)] = :back
constraints[T.contact_addr(2, :x)] = 0.0
constraints[T.contact_addr(2, :y)] = 0.0


original_trace, _ = Gen.generate(T.scene, (params,), constraints);
Gen.get_choices(original_trace)

slack_offset_sweep = 0.0:0.02:0.5
slack_dir_angle_sweep = 0.0:0.02:pi/10

traces = fill(Any, (size(slack_offset_sweep)[1], size(slack_dir_angle_sweep)[1]))

traces = [
    let 
        slack_dir = GDS.UnitVector3([0.0, sin(slack_dir_angle), cos(slack_dir_angle)])

        slack_constraints = Gen.choicemap()

        slack_constraints[T.contact_addr(2,:slack_offset)] = slack_offset
        slack_constraints[T.contact_addr(2,:slack_dir)] = slack_dir

        trace, = Gen.update(original_trace, slack_constraints)
        trace
    end
    for (i, slack_offset) in enumerate(slack_offset_sweep),
        (j, slack_dir_angle) in enumerate(slack_dir_angle_sweep)
];


## Get Poses with T.get_poses(traces[1,1])
## Setup a new renderer in color mode
## Add the two box meshes (using GL.box_mesh_from_dims)
## Rendering them in red and blue

## rgb_image, dpeth_image = gl_render (...)
## x = GL.view_rgb_image(rgb_image)

images = fill(Any, size(traces))

##

## Involutive MCMC (toggle edge moves).
##  for _ in 1:1000
##     trace, acc = T.toggle_edge_move(trace, 1, 2);
## Count fraction where graph has an edge.
