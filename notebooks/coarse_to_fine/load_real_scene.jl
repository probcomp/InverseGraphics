import Pkg; Pkg.status()

import InverseGraphics as T
import Open3DVisualizer as V

camera_intrinsics = T.GL.CameraIntrinsics(
	640, 480,
	507.8159715323307,
	508.9675493512412,
	307.0591715498334,
	231.8315010235393,
	0.01,
	10000.0
)

mesh_names = ["cube_triangulated.obj",
              "sphere_triangulated.obj",
              "cylinder_small_triangulated.obj",
              "triangular_prism_triangulated.obj",
              "crescent_triangulated.obj",
             ]

meshdir = joinpath(dirname(dirname(pathof(T))),"notebooks/coarse_to_fine/shape_models");
meshes = [
    T.GL.get_mesh_data_from_obj_file(joinpath(meshdir, m))
    for m in mesh_names
];

data_path = joinpath(dirname(dirname(pathof(T))),"notebooks/coarse_to_fine/ibm_data")
IDX = 218
rgb_image = T.load_rgb(joinpath(data_path, "color_$(lpad(IDX,4,"0")).png"));

d = T.load_depth(joinpath(data_path, "depth_$(lpad(IDX,4,"0")).png"));
d = d ./ 5.0


c = T.GL.depth_image_to_point_cloud(d, camera_intrinsics);
c = c[:,c[3,:] .> 1e-4]
c = c[:, c[3,:] .< 200.1]

best_eq, sub_cloud, _ = T.find_plane(c;threshold=0.01);
cam_pose = T.camera_pose_from_table_eq(best_eq);
c = T.move_points_to_frame_b(c, cam_pose);
c = c[:, c[3,:] .> 1.0]

assign =T.dbscan_cluster(c; radius=1.0, min_cluster_size=50);
entities = T.get_entities_from_assignment(c, assign);
@show length(entities)

V.open_window()
e = entities[4]
V.clear()
V.add(V.make_point_cloud(c ;color = T.I.colorant"red"));
V.add(V.make_point_cloud(e ;color = T.I.colorant"blue"));
V.sync()
V.run()


