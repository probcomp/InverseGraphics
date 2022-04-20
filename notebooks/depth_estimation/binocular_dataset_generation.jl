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
import Random
V.setup_visualizer()


meshdir = joinpath(dirname(dirname(pathof(T))),"notebooks/depth_estimation")
mesh_names = ["cube_triangulated.obj",
              "sphere_triangulated.obj",
              "cylinder_small_triangulated.obj",
              "triangular_prism_triangulated.obj",
              "crescent_triangulated.obj",
              "arch_triangulated.obj"
             ]
let 
    meshes = [
        GL.get_mesh_data_from_obj_file(joinpath(meshdir, m))
        for m in mesh_names
    ];
    meshes = [
        T.scale_mesh(m, 0.1)
        for m in meshes
    ]
    bbox_and_poses = (x -> T.axis_aligned_bounding_box(x.vertices)).(meshes)
    for i in 1:length(meshes)
        meshes[i].vertices = meshes[i].vertices .- bbox_and_poses[i].pose.pos
    end
    [i.box for i in bbox_and_poses], meshes
end




table_bbox = S.Box(10.0, 0.1, 10.0)
table_mesh = GL.box_mesh_from_dims(T.get_dims(table_bbox))


background_bbox = S.Box(100.0, 100.0, 0.1)
background_mesh = GL.box_mesh_from_dims(T.get_dims(background_bbox))


# Initialize the canera intrinsics and renderer that will render using those intrinsics.
camera = GL.CameraIntrinsics()
renderer = GL.setup_renderer(camera, GL.RGBMode(), gl_version=(3,3))
GL.load_object!(renderer, table_mesh)
GL.load_object!(renderer, background_mesh)
for m in meshes
    GL.load_object!(renderer, m)
end

function generate_binocular_pair()
    scene_graph = S.SceneGraph()
    S.addObject!(scene_graph, :table, table_bbox)
    S.addObject!(scene_graph, :background, background_bbox)
    for (idx, bbox) in enumerate(boxes)
        # Each Shape object has an associated bounding box.
        S.addObject!(scene_graph, T.obj_name_from_idx(idx), bbox)
    end
    S.setPose!(scene_graph, :table, Pose([1.0, 6.0, 16.0]))
    S.setPose!(scene_graph, :background, Pose([0.0, 0.0, 30.0]))

    for (idx, bbox) in enumerate(boxes)
        x,y = Gen.uniform(-5.0, 5.0), Gen.uniform(-5.0, 5.0)
        ang = Gen.uniform(-5.0, 5.0)
        face = S.BOX_SURFACE_IDS[Gen.categorical(ones(length(S.BOX_SURFACE_IDS)) ./ length(S.BOX_SURFACE_IDS))]
        contact = S.ShapeContact(:back, Float64[], face, Float64[], S.PlanarContact(x, y, ang))
        S.setContact!(scene_graph, :table, T.obj_name_from_idx(idx), contact)
    end

    obj_colors = [I.colorant"red",I.colorant"green", I.colorant"blue",I.colorant"yellow",I.colorant"orange"]
    colors = [I.colorant"tan", I.colorant"grey", Random.shuffle(obj_colors)...]

    poses = T.floatingPosesOf(scene_graph)
    cam_pose = Pose(zeros(3), R.RotX(-pi/8));
    rgb_image_1, depth_image_1 = GL.gl_render(renderer, collect(1:length(poses)), poses, cam_pose; colors=colors);
    baseline_transform = Pose(2.0, 0.0, 0.0)
    rgb_image_2, _ = GL.gl_render(renderer, collect(1:length(poses)), poses, cam_pose * baseline_transform; colors=colors);
    rgb_image_1, rgb_image_2, depth_image_1
end

img_1, img_2, depth_image_1 = generate_binocular_pair() 
IV.imshow(GL.view_rgb_image(img_1));
IV.imshow(GL.view_rgb_image(img_2));

try
    mkdir("dataset")
catch
end

import FileIO
import Serialization
import PyCall
np = PyCall.pyimport("numpy")
for t in 1:5000
    img_1, img_2, depth_1 = generate_binocular_pair() 
    img_1, img_2 = GL.view_rgb_image.([img_1, img_2])
    file_number = lpad(t,5,"0")
    FileIO.save("dataset/$(file_number)_left.png", img_1)
    FileIO.save("dataset/$(file_number)_right.png", img_2)
    Serialization.serialize("dataset/$(file_number)_left_depth_julia", depth_1)
    np.save("dataset/$(file_number)_left_depth_numpy", depth_1)
end

Serialization.serialize("dataset/camera_intrinsics", camera)
open("dataset/K.txt","a") do io
    println(io, camera)
end


renderer = GL.setup_renderer(camera, GL.TextureMode())