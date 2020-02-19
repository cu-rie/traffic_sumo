import numpy
import dgl
import itertools
import torch

NUM_NODES = 4
horizontal_nodes = [0, 2]
vertical_nodes = [1, 3]
SAME_EDGE_TYPE = 0
DIFF_EDGE_TYPE = 1
NUM_EDGE_TYPES = 2


def state2graphfunc(state, device=None):
    g = dgl.DGLGraph()

    # generate node with node feature
    state = torch.Tensor(state).reshape(-1, 1).to(device)
    g.add_nodes(NUM_NODES, {'init_node_feature': state})

    # add edge connection with same signal nodes
    edge_same_1 = cartesian_product(horizontal_nodes, horizontal_nodes)
    edge_same_2 = cartesian_product(vertical_nodes, vertical_nodes)

    edge_same_type = torch.Tensor(data=(SAME_EDGE_TYPE,)).to(device)
    edge_same_one_hot = torch.Tensor(get_one_hot_edge_type(SAME_EDGE_TYPE)).to(device)

    num_same_edges_1 = len(edge_same_1[0])
    num_same_edges_2 = len(edge_same_2[0])

    g.add_edges(edge_same_1[0], edge_same_1[1], {'edge_type': edge_same_type.repeat(num_same_edges_1, 1),
                                                 'edge_one_hot': edge_same_one_hot.repeat(num_same_edges_1, 1)})

    g.add_edges(edge_same_2[0], edge_same_2[1], {'edge_type': edge_same_type.repeat(num_same_edges_2, 1),
                                                 'edge_one_hot': edge_same_one_hot.repeat(num_same_edges_2, 1)})

    # add edge connection with different signal nodes
    edge_diff_1 = cartesian_product(horizontal_nodes, vertical_nodes)
    edge_diff_2 = cartesian_product(vertical_nodes, horizontal_nodes)

    num_diff_edges_1 = len(edge_diff_1[0])
    num_diff_edges_2 = len(edge_diff_2[0])

    edge_diff_type = torch.Tensor(data=(DIFF_EDGE_TYPE,)).to(device)
    edge_diff_one_hot = torch.Tensor(get_one_hot_edge_type(DIFF_EDGE_TYPE)).to(device)

    g.add_edges(edge_diff_1[0], edge_diff_1[1], {'edge_type': edge_diff_type.repeat(num_diff_edges_1, 1),
                                                 'edge_one_hot': edge_diff_one_hot.repeat(num_diff_edges_1, 1)})

    g.add_edges(edge_diff_2[0], edge_diff_2[1], {'edge_type': edge_diff_type.repeat(num_diff_edges_2, 1),
                                                 'edge_one_hot': edge_diff_one_hot.repeat(num_diff_edges_2, 1)})

    return g


def cartesian_product(*iterables, return_1d=False, self_edge=False):
    if return_1d:
        xs = []
        ys = []
        if self_edge:
            for ij in itertools.product(*iterables):
                xs.append(ij[0])
                ys.append(ij[1])
        else:
            for ij in itertools.product(*iterables):
                if ij[0] != ij[1]:
                    xs.append(ij[0])
                    ys.append(ij[1])
        ret = (xs, ys)
    else:
        ret = [i for i in itertools.product(*iterables)]
    return ret


def get_one_hot_edge_type(edge_type: int):
    ret = [0] * NUM_EDGE_TYPES
    ret[edge_type] = 1.0
    return ret
