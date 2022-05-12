import InverseGraphics as T
import Gen
import NearestNeighbors as NN


X = rand(3,10000) * 20.0
Y = rand(3,5000) * 20.0

tree = NN.KDTree(X)

@time NN.nn(tree, Y);
r = 1.0
@time all_idxs = NN.inrange(tree, Y, r);