import BenchmarkTools: @benchmark

function voxelize(cloud, resolution)
    cloud_xyz = round.(cloud[1:min(size(cloud,1), 3),:] ./ resolution) * resolution
    idxs = unique(i -> cloud_xyz[:,i], 1:size(cloud_xyz)[2])
    cloud_xyz[:, idxs]
end

function voxelize_2(cloud, resolution)
    cloud_xyz = round.(UInt8, cloud[1:min(size(cloud,1), 3),:] ./ resolution)
    hcat(Set([collect(eachcol(cloud_xyz))]...)...)
end

workspace = falses(1000, 1000, 1000);
function voxelize_3(cloud, resolution)
	workspace[:] .= false
    cloud_xyz = round.(UInt8, cloud[1:min(size(cloud,1), 3),:] ./ resolution)
    for (x,y,z) in eachcol(cloud_xyz)
    	workspace[x+1, y+1, z+1] = true
    end
    hcat( [[Tuple(x)...] for x in findall(workspace)]...)
end


cloud = rand(3, 100000) * 50.0
resolution = 0.5

@time data = voxelize(cloud, resolution)
@time data = voxelize_2(cloud, resolution)
@time data = voxelize_3(cloud, resolution)

@benchmark voxelize(cloud, resolution)
@benchmark voxelize_2(cloud, resolution)