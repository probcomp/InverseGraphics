import Open3DVisualizer as V
import PoseComposition: Pose, IDENTITY_POSE, IDENTITY_ORN
import Rotations as R
V.open_window()

a1 = V.make_axes(1.0)
a1 = V.move_mesh_to_pose(a1, IDENTITY_POSE)
V.add(a1)

a2 = V.make_axes(0.5)
a2 = V.move_mesh_to_pose(a2, Pose(ones(3), R.RotXYZ(0.1, 0.4, -2.0)))
V.add(a2)

a3 = V.make_axes(0.1)
a3 = V.move_mesh_to_pose(a3, Pose(-2.0 * ones(3), R.RotXYZ(-1.1, 0.4, -1.0)))
V.add(a3)

V.run()
