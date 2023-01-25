import numpy as np
import torch
import torch_geometric


def get_batch(adj_label, idx_train, features, edge_index, labels, batch_size=2000, sampler='random_batch', cuda=True):

    # if batch_size is smaller than len(idx_train), remove everything except idx_train
    if batch_size < len(idx_train):
        adj_label = adj_label[idx_train, :][:, idx_train]
        features = features[idx_train]
        labels = labels[idx_train]
        edge_index = edge_index[:, torch.isin(edge_index[0], idx_train) & torch.isin(edge_index[1], idx_train)]
        idx_train = torch.tensor(list(range(0, batch_size)))

    if sampler == 'random_batch':
        return random_batch(adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'random_pagerank':
        return random_pagerank(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'random_degree':
        return random_degree(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'rank_degree':
        return rank_degree(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'list':
        return list_sampling(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'negative':
        return negative_sampling(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'random_edge':
        return random_edge(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'random_node_edge':
        return random_node_edge(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'hybrid_edge':
        return hybrid_edge(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'fixed_size_neighbor':
        return fixed_size_neighbor(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'random_node_neighbor':
        return random_node_neighbor(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'random_walk':
        return random_walk(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'random_jump':
        return random_jump(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'forest_fire':
        return forest_fire(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'frontier':
        return frontier(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)
    elif sampler == 'snowball':
        return snowball(edge_index, adj_label, idx_train, features, labels, batch_size, cuda)


def idx_to_adj(node_index, idx_train, adj_label, features, labels, batch_size):
    if len(idx_train) < batch_size:
        node_index[0:len(idx_train)] = idx_train
        new_idx = list(range(0, len(idx_train)))
    else:
        new_idx = list(range(0, batch_size))
    features_batch = features[node_index]
    adj_label_batch = adj_label[node_index, :][:, node_index]
    labels_batch = labels[node_index]
    return features_batch, adj_label_batch, labels_batch, new_idx


def random_batch(adj_label, idx_train, features, labels, batch_size, cuda):
    """
        get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long)
    if cuda:
        rand_indx = rand_indx.cuda()
    rand_indx[0:len(idx_train)] = idx_train

    return idx_to_adj(rand_indx, idx_train, adj_label, features, labels, batch_size)


def random_pagerank(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Fabi
    pass


def random_degree(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Tobi
    pass


def rank_degree(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Jan
    pass


def list_sampling(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Jan
    pass


def negative_sampling(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    device = torch.device('cuda' if cuda else 'cpu')
    # new edge index = all not existing edges
    new_edge_index = torch_geometric.utils.negative_sampling(edge_index)
    new_edge_index = new_edge_index.to(device)
    # select random batch_size edges
    chosen_edges = torch.tensor(np.random.choice(np.arange(new_edge_index.shape[1]), batch_size)).type(torch.long).to(
        device)
    chosen_nodes = torch.unique(new_edge_index[:, chosen_edges]).to(device)

    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size)


def random_edge(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Tobi
    pass


def random_node_edge(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Tobi
    pass


def hybrid_edge(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Tobi
    pass


def fixed_size_neighbor(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Jan
    pass


def random_node_neighbor(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # we select a node uniformly at random together with all of its out-going neighbors.
    device = torch.device('cuda' if cuda else 'cpu')
    # empty tensor for new nodes
    chosen_nodes = torch.tensor([]).type(torch.long).to(device)
    edge_index = edge_index.to(device)
    while True:
        chosen_node = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), 1)).type(torch.long).to(device)
        outgoing_nodes = edge_index[1][edge_index[0] == chosen_node]
        chosen_nodes = torch.cat((chosen_nodes, chosen_node, outgoing_nodes)).to(device)
        if chosen_nodes.shape[0] >= batch_size:
            break
    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size)

def random_walk(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Jan
    pass


def random_jump(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Jan
    pass


def forest_fire(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Tobi
    pass


def frontier(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # fixme @Fabi
    device = torch.device('cuda' if cuda else 'cpu')
    chosen_nodes = torch.tensor([]).type(torch.long).to(device)
    m = 10 # TODO tweak parameter and mention in section 3
    # init L with m randomly chosen nodes (uniformly)
    L = np.random.choice(np.arange(adj_label.shape[0]), m)
    while True:
        # calculate the degree of each node in L
        degrees = np.array([edge_index[0][edge_index[1] == node].shape[0] for node in L])
        sum_of_degrees = degrees.sum()
        # select randome node u from L with probability degree(u)/sum_v in L degree(v)
        u = np.random.choice(L, 1, p=[d/sum_of_degrees for d in degrees])
        # select random neighbor v of u
        outgoing_nodes = edge_index[1][edge_index[0] == u]
        # randomly choose one of the neighbors
        rand_idx = np.random.choice(np.arange(outgoing_nodes.shape[0]), 1)
        v = outgoing_nodes[rand_idx]
        # replace u with v in L
        L = np.where(L == u, v, L)
        # add u and v to chosen nodes
        chosen_nodes = torch.unique(torch.cat((chosen_nodes, torch.tensor([u, v]).type(torch.long).to(device))))
        if chosen_nodes.shape[0] >= batch_size:
            break



def snowball(edge_index, adj_label, idx_train, features, labels, batch_size, cuda):
    # TODO @Fabi
    pass
