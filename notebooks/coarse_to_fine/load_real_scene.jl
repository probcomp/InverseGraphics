import InverseGraphics as T
import Open3DVisualizer as V

V.setup_visualizer()
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
IDX = 345
rgb_image = T.load_rgb(joinpath(data_path, "color_$(lpad(IDX,4,"0")).png"));
IV.imshow(T.GL.view_rgb_image(rgb_image;in_255=true))

d = d ./ 1000.0

d = T.load_depth(joinpath(data_path, "depth_$(lpad(IDX,4,"0")).png"));
d = d ./ 1000.0

c = T.GL.depth_image_to_point_cloud(d, camera_intrinsics);
c = c[:,c[3,:] .> 1e-4]
c = c[:, c[3,:] .< 1.1]

V.reset_visualizer()
V.viz(c);

best_eq, sub_cloud, _ = T.find_plane(c;threshold=0.01);

V.reset_visualizer()
V.viz(c);
V.viz(sub_cloud;channel_name=:obs, color=T.I.colorant"red")

cam_pose = T.camera_pose_from_table_eq(best_eq);
c = T.move_points_to_frame_b(c, cam_pose);

V.reset_visualizer()
V.viz(c);

c = c[:, c[3,:] .> 0.01]
V.reset_visualizer()
V.viz(c);


assign =T.dbscan_cluster(c; radius=0.03, min_cluster_size=50);
entities = T.get_entities_from_assignment(c, assign);
@show length(entities)


colors = T.I.distinguishable_colors(lenth(entities))
X.open_window()
for (col,e) in zip(colors, entities)
	X.add(X.make_point_cloud(e;color=col))
end
X.run()

