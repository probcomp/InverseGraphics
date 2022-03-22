module InverseGraphics

using Gen

import GLRenderer as GL
import GenDirectionalStats
import MiniGSG as S
import Rotations as R
import PoseComposition as PC
import PoseComposition: Pose
import Colors
import PyCall
import LightGraphs as LG
import MetaGraphs as MG
import StaticArrays: SVector, @SVector, StaticVector

function __init__()
    numpy = PyCall.pyimport("numpy")
    nothing
end

include("utils/bbox.jl")
include("utils/clustering.jl")
include("utils/data_loader.jl")
include("utils/icp.jl")
include("utils/occlusions.jl")
include("utils/pose.jl")
include("utils/pose_utils.jl")
include("utils/scale_camera.jl")
include("utils/utils.jl")
include("utils/table_detection.jl")

include("distributions/distributions.jl")

include("contact/contact.jl")
include("graphs/graphs.jl")
include("shape/shape.jl")

include("model/sg_model.jl")
include("model/sg_model_utils.jl")

include("inference/inference.jl")

include("feature_detection/feature_detection.jl")

include("mcs/mcs_utils.jl")

end # module
