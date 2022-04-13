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
import PyCall
import Serialization
try
    import MeshCatViz as V
catch
    import MeshCatViz as V    
end
import Random
V.setup_visualizer()


camera = GL.CameraIntrinsics()


PyCall.py"""
import cv2
import tensorflow as tf
import numpy as np

from hitnet import HitNet, ModelType, draw_disparity, draw_depth, CameraConfig, load_img, load_img_from_file

# Select model type
# model_type = ModelType.middlebury
model_type = ModelType.flyingthings
# model_type = ModelType.eth3d

if model_type == ModelType.middlebury:
	model_path = "models/middlebury_d400.pb"
elif model_type == ModelType.flyingthings:
	model_path = "models/flyingthings_finalpass_xl.pb"
elif model_type == ModelType.eth3d:
	model_path = "models/eth3d.pb"
hitnet_depth = HitNet(model_path, model_type)

def infer_disparity_map(left_img_filename, right_img_filename):
	left_img = load_img_from_file(left_img_filename)
	right_img = load_img_from_file(right_img_filename)
	disparity_map = hitnet_depth(left_img, right_img)
	return disparity_map
"""

dataset_dir = "/home/nishadg/mit/InverseGraphics/dataset"
t = 2

actual_depth = Serialization.deserialize(joinpath(dataset_dir, "$(lpad(t,5,'0'))_left_depth_julia"))
actual_cloud = GL.depth_image_to_point_cloud(actual_depth, camera)

disparity_map = PyCall.py"infer_disparity_map"(
	joinpath(dataset_dir, "$(lpad(t,5,'0'))_left.png"),
	joinpath(dataset_dir, "$(lpad(t,5,'0'))_right.png")
)

depth = 2.0 * camera.fx ./ disparity_map 
cloud = GL.depth_image_to_point_cloud(depth, camera)
IV.imshow(GL.view_depth_image(depth))
V.reset_visualizer()
V.viz(cloud ./ 10.0; channel_name=:inf, color=I.colorant"blue")
V.viz(actual_cloud./ 10.0; channel_name=:gen, color=I.colorant"red")


import ColorSchemes

IV.imshow(get(ColorSchemes.blackbody, depth, (0.0, 30.0)))	
IV.imshow(get(ColorSchemes.blackbody, actual_depth, (0.0, 30.0)))	

import FileIO
FileIO.save("inferred_depth.png", get(ColorSchemes.blackbody, depth, (0.0, 30.0)))
FileIO.save("actual_depth.png", get(ColorSchemes.blackbody, actual_depth, (0.0, 30.0)))


# V.viz(actual_cloud;channel_name=:real, color=I.colorant"black")


table_eq = T.find_table_plane(cloud)
inliers ,mask = T.find_plane_inliers(cloud, table_eq);
V.reset_visualizer()
V.viz(cloud ./ 10.0; channel_name=:inf, color=I.colorant"blue")
V.viz(inliers./ 10.0; channel_name=:gen, color=I.colorant"black")
