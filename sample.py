import logging
import random

import numpy as np
import torch
import torch_geometric


def get_batch(adj_label, idx_train, features, edge_index, batch_size=2000, sampler='random_batch', cuda=True):

    # if batch_size is smaller than len(idx_train), remove everything except idx_train
    if batch_size < len(idx_train):
        adj_label = adj_label[idx_train, :][:, idx_train]
        features = features[idx_train]
        edge_index = edge_index[:, edge_index[0].isin(idx_train) & edge_index[1].isin(idx_train)]
        idx_train = list(range(0, len(idx_train)))

    if sampler == 'random_batch':
        return random_batch(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'random_pagerank':
        return random_pagerank(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'random_degree':
        return random_degree(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'rank_degree':
        return rank_degree(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'list':
        return list_sampling(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'negative':
        return negative_sampling(edge_index, adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'random_edge':
        return random_edge(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'random_node_edge':
        return random_node_edge(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'hybrid_edge':
        return hybrid_edge(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'fixed_size_neighbor':
        return fixed_size_neighbor(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'random_node_neighbor':
        return random_node_neighbor(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'random_walk':
        return random_walk(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'random_jump':
        return random_jump(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'forest_fire':
        return forest_fire(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'frontier':
        return frontier(adj_label, idx_train, features, batch_size, cuda)
    elif sampler == 'snowball':
        return snowball(adj_label, idx_train, features, batch_size, cuda)


def idx_to_adj(node_index, idx_train, adj_label, features, batch_size):
    if len(idx_train) < batch_size:
        node_index[0:len(idx_train)] = idx_train
        new_idx = list(range(0, len(idx_train)))
    else:
        new_idx = list(range(0, batch_size))
    features_batch = features[node_index]
    adj_label_batch = adj_label[node_index, :][:, node_index]
    return features_batch, adj_label_batch, new_idx


def random_batch(adj_label, idx_train, features, batch_size, cuda):
    """
        get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long)
    if cuda:
        rand_indx = rand_indx.cuda()
    rand_indx[0:len(idx_train)] = idx_train

    return idx_to_adj(rand_indx, idx_train, adj_label, features, batch_size)


def random_pagerank(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Fabi
    pass


def random_degree(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Tobi
    pass


def rank_degree(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Jan
    pass


def list_sampling(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Jan
    pass


def negative_sampling(edge_index, adj_label, idx_train, features, batch_size, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    # new edge index = all not existing edges
    new_edge_index = torch_geometric.utils.negative_sampling(edge_index)
    new_edge_index = new_edge_index.to(device)
    # select random batch_size edges
    chosen_edges = torch.tensor(np.random.choice(np.arange(new_edge_index.shape[1]), batch_size)).type(torch.long).to(
        device)
    chosen_nodes = torch.unique(new_edge_index[:, chosen_edges]).to(device)

    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, batch_size)


def random_edge(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Tobi
    pass


def random_node_edge(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Tobi
    pass


def hybrid_edge(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Tobi
    pass


def fixed_size_neighbor(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Jan
    pass


def random_node_neighbor(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Fabi
    pass


def random_walk(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Jan
    pass


def random_jump(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Jan
    pass


def forest_fire(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Tobi
    pass


def frontier(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Fabi
    pass


def snowball(adj_label, idx_train, features, batch_size, cuda):
    # TODO @Fabi
    pass
