import Pkg; Pkg.status()

import InverseGraphics as T
import Open3DVisualizer as V
import MeshCatViz as MV

MV.setup_visualizer()

camera_intrinsics = T.GL.CameraIntrinsics(
	640, 480,
	507.8159715323307,
	508.9675493512412,
	307.0591715498334,
	231.8315010235393,
	0.01,
	10000.0
)

mesh_names = [
	"cube_triangulated.obj",
	"arch_triangulated.obj",
	"sphere_triangulated.obj",
	"cylinder_small_triangulated.obj",
	"triangular_prism_triangulated.obj",
	"cuboid_triangulated.obj",
	"crescent_triangulated.obj",
]

meshdir = joinpath(dirname(dirname(pathof(T))),"notebooks/coarse_to_fine/shape_models");
meshes = [
    T.GL.get_mesh_data_from_obj_file(joinpath(meshdir, m))
    for m in mesh_names
];

data_path = joinpath(dirname(dirname(pathof(T))),"notebooks/coarse_to_fine/ibm_data")
IDX = 216
rgb_image = T.load_rgb(joinpath(data_path, "color_$(lpad(IDX,4,"0")).png"));

d = T.load_depth(joinpath(data_path, "depth_$(lpad(IDX,4,"0")).png"));


import PyCall
np = PyCall.pyimport("numpy")
d = np.load(joinpath(dirname(dirname(pathof(T))),"notebooks/coarse_to_fine/depth_0001.npy"));


c = T.GL.depth_image_to_point_cloud(d, camera_intrinsics);
c = c[:,c[3,:] .> 1e-4]
c = c[:, c[3,:] .< 1.4]

MV.reset_visualizer()
MV.viz(c;channel_name=:x,color=T.I.colorant"red")


best_eq, sub_cloud, _ = T.find_plane(c;threshold=0.01);
cam_pose = T.camera_pose_from_table_eq(best_eq);
c = T.move_points_to_frame_b(c, cam_pose);

c = c[:, c[3,:] .> 0.01]

assign =T.dbscan_cluster(c; radius=0.01, min_cluster_size=50);
entities = T.get_entities_from_assignment(c, assign);
@show length(entities)

MV.reset_visualizer()
e =entities[4] 
MV.viz(c;channel_name=:x,color=T.I.colorant"red")
MV.viz(e;channel_name=:t,color=T.I.colorant"blue")

V.open_window()

V.clear()
V.add(V.make_point_cloud(e .- T.centroid(e);color = T.I.colorant"red"));
V.add(V.make_point_cloud(meshes[1].vertices / 200.0 ;color = T.I.colorant"blue"));
V.add(V.make_axes(0.01))
V.sync()

V.run()


V.open_window()

V.clear()
V.add(V.make_point_cloud(c;color = T.I.colorant"red"));
V.add(V.make_point_cloud(e ;color = T.I.colorant"blue"));
V.sync()


