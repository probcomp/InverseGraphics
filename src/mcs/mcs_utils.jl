function CameraIntrinsics(step_metadata::PyCall.PyObject)::GL.CameraIntrinsics
    width, height = step_metadata.camera_aspect_ratio
    aspect_ratio = width / height

    # Camera principal point is the center of the image.
    cx, cy = width / 2.0, height / 2.0

    # Vertical field of view is given.
    fov_y = deg2rad(step_metadata.camera_field_of_view)
    # Convert field of view to distance to scale by aspect ratio and
    # convert back to radians to recover the horizontal field of view.
    fov_x = 2 * atan(aspect_ratio * tan(fov_y / 2.0))

    # Use the following relation to recover the focal length:
    #   FOV = 2 * atan( (0.5 * IMAGE_PLANE_SIZE) / FOCAL_LENGTH )
    fx = cx / tan(fov_x / 2.0)
    fy = cy / tan(fov_y / 2.0)

    clipping_near, clipping_far = step_metadata.camera_clipping_planes

    GL.CameraIntrinsics(width, height,
        fx, fy, cx, cy,
        clipping_near, clipping_far)
end

get_depth_image_from_step_metadata(step_metadata::PyCall.PyObject) = Matrix(last(step_metadata.depth_map_list))
get_rgb_image_from_step_metadata(step_metadata::PyCall.PyObject) = Float64.(numpy.array(last(step_metadata.image_list)))
