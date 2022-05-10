import Pkg
using Revise
Pkg.add("ProgressBars")
using ProgressBars

ENV["PYTHON"] = "/usr/bin/python3"
Pkg.build("PyCall")

using Gen

import GLRenderer as GL
using Images
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

function Gen.random(
    ::T.UniformMixtureFromTemplateMultiCloud, X::Array{Matrix{Float64}},
    p_outlier::Float64, radius::Float64, bounds::Tuple)
    X[1]
end

constraints = Gen.choicemap()
g = T.graph_with_edges(2, [1=>2])
# 1=>2
constraints[T.structure_addr()] = g
constraints[T.floating_pose_addr(1)] = IDENTITY_POSE
constraints[:p_outlier] = hypers.p_outlier
constraints[:noise] = hypers.noise
constraints[T.contact_addr(2, :child_face)] = :back
constraints[T.contact_addr(2, :parent_face)] = :back
constraints[T.contact_addr(2, :x)] = 0.0
constraints[T.contact_addr(2, :y)] = 0.0

@gen (static) function scene_no_likelihood(model_params::T.SceneModelParameters)
    camera_pose ~ T.uniformPose(-1000.0,1000.0,-1000.0,1000.0,-1000.0,1000.0)
    scene_graph ~ T.scene_graph_prior(model_params)
    return (scene_graph=scene_graph)
end

original_trace, _ = Gen.generate(scene_no_likelihood, (params,), constraints);
Gen.get_choices(original_trace)

slack_offset_sweep = 0.0:0.02:0.5
slack_dir_angle_sweep = 0.0:0.02:pi/10

traces = fill(Any, (size(slack_offset_sweep)[1], size(slack_dir_angle_sweep)[1]))

traces = [
    let 
        slack_dir = GDS.UnitVector3([0.0, sin(slack_dir_angle), -cos(slack_dir_angle)])

        slack_constraints = Gen.choicemap()

        slack_constraints[T.contact_addr(2,:slack_offset)] = slack_offset
        slack_constraints[T.contact_addr(2,:slack_dir)] = slack_dir

        trace, = Gen.update(original_trace, slack_constraints)
        trace
    end
    for (i, slack_offset) in enumerate(slack_offset_sweep),
        (j, slack_dir_angle) in enumerate(slack_dir_angle_sweep)
];

images = fill(zeros(120, 160, 4), size(traces))

rgb_renderer = GL.setup_renderer(camera, GL.RGBBasicMode())

for box in object_boxes
    mesh = GL.box_mesh_from_dims(T.get_dims(box))
    GL.load_object!(rgb_renderer, mesh)
end


for i in eachindex(traces)
    trace = traces[i]

    rgb_image, depth_image = GL.gl_render(
        rgb_renderer, ids, T.get_poses(trace), 
        Pose(0, 0, -5); colors=[I.colorant"red", I.colorant"blue"],
    )

    images[i] = rgb_image
end

proportion_with_edge = zeros(size(traces))

for i in ProgressBar(eachindex(traces))
    trace = traces[i]
    
    num_with_edges = 0.0
    for _ in 1:20
        trace, acc = T.toggle_edge_move(trace, 2, 1);
        
        if length(T.get_edges(trace)) == 1
            num_with_edges += 1.0
        end
    end
    
    proportion_with_edge[i] = num_with_edges / 2.0
    traces[i] = trace
end

proportion_with_edge

Pkg.add("CairoMakie")
import CairoMakie
CairoMakie.heatmap(proportion_with_edge)

length(slack_offset_sweep), length(slack_dir_angle_sweep)

slack_offset_sweep

CairoMakie.heatmap(slack_offset_sweep, slack_dir_angle_sweep, proportion_with_edge)

#  = 0.0:0.02:0.5
#  = 0.0:0.02:pi/10
