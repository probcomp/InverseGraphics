function are_features_visible(coords, depth_image, camera_intrinsics, camera_pose; window_half_width=3)
    coords_in_camera_frame = get_points_in_frame_b(coords, camera_pose)
    pixels_y_x = [(y,x) for (x,y) in eachcol(GL.point_cloud_to_pixel_coordinates(coords_in_camera_frame, camera_intrinsics))]
    valid = fill(false, size(coords_in_camera_frame)[2])
    for i in 1:length(valid)
        (y,x) = pixels_y_x[i]
        if !((1 <= y <= camera_intrinsics.height) && (1<= x <= camera_intrinsics.width))
           continue 
        end
        
        depth_vals = depth_image[max(1,y - window_half_width)  : min(y + window_half_width, camera_intrinsics.height) 
            , max(1,x - window_half_width) : min(x + window_half_width , camera_intrinsics.width)
        ]
        expected_depth_val = coords_in_camera_frame[3,i]
        good = any( abs.(depth_vals .- expected_depth_val) .< 0.2)
        valid[i] = good
    end
    valid    
end
