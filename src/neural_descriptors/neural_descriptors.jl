import NearestNeighbors as NN


function get_neural_descriptor_score(renderer, tree, id, p, query_img_flattened, query_img_size, obj_keys, log_denominator, intrinsics)
    depth_image = GL.gl_render(renderer, [id], [p], IDENTITY_POSE)
    cloud = GL.depth_image_to_point_cloud(depth_image, renderer.camera_intrinsics);
    if isnothing(cloud)
        return -Inf
    end
    cloud_in_object_reference_frame = GL.get_points_in_frame_b(cloud, p);
    
    idxs, _ = NN.nn(tree, cloud_in_object_reference_frame)
    model_point_keys = obj_keys[:, idxs]

    model_point_pixel_coords = GL.point_cloud_to_pixel_coordinates(cloud, intrinsics);
    
    flattened_pixel_coords = (model_point_pixel_coords[1,:] .- 1) .* query_img_size[1] .+ model_point_pixel_coords[2,:] 
    
    model_point_corresponding_obs_embedding = query_img_flattened[:, flattened_pixel_coords]
    
    inner_prod = sum(model_point_corresponding_obs_embedding .* model_point_keys; dims=1)[:]
    model_point_log_denominators = [log_denominator[y,x] for (x,y) in eachcol(model_point_pixel_coords)]
    log_probs = inner_prod - model_point_log_denominators 

    num_unexplained_pixels = (size(query_img_flattened)[2] - length(log_probs))
    
    log_score = log(1.0 / size(obj_keys,2)) * num_unexplained_pixels + sum(log_probs)
    log_score, log_probs
end
