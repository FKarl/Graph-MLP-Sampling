import numpy as np
import torch


def get_batch(adj_label, idx_train, features, batch_size=2000, sampler='random_batch'):
    if sampler == 'random_batch':
        return random_batch(adj_label, idx_train, features, batch_size)
    elif sampler == 'random_pagerank':
        return random_pagerank(adj_label, idx_train, features, batch_size)
    elif sampler == 'random_degree':
        return random_degree(adj_label, idx_train, features, batch_size)
    elif sampler == 'rank_degree':
        return rank_degree(adj_label, idx_train, features, batch_size)
    elif sampler == 'list':
        return list_sampling(adj_label, idx_train, features, batch_size)
    elif sampler == 'negative':
        return negative_sampling(adj_label, idx_train, features, batch_size)
    elif sampler == 'random_edge':
        return random_edge(adj_label, idx_train, features, batch_size)
    elif sampler == 'random_node_edge':
        return random_node_edge(adj_label, idx_train, features, batch_size)
    elif sampler == 'hybrid_edge':
        return hybrid_edge(adj_label, idx_train, features, batch_size)
    elif sampler == 'fixed_size_neighbor':
        return fixed_size_neighbor(adj_label, idx_train, features, batch_size)
    elif sampler == 'random_node_neighbor':
        return random_node_neighbor(adj_label, idx_train, features, batch_size)
    elif sampler == 'random_walk':
        return random_walk(adj_label, idx_train, features, batch_size)
    elif sampler == 'random_jump':
        return random_jump(adj_label, idx_train, features, batch_size)
    elif sampler == 'forest_fire':
        return forest_fire(adj_label, idx_train, features, batch_size)
    elif sampler == 'frontier':
        return frontier(adj_label, idx_train, features, batch_size)
    elif sampler == 'snowball':
        return snowball(adj_label, idx_train, features, batch_size)


def random_batch(adj_label, idx_train, features, batch_size=2000):
    """
        get a batch of feature & adjacency matrix
        """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long).cuda()
    rand_indx[0:len(idx_train)] = idx_train
    features_batch = features[rand_indx]
    adj_label_batch = adj_label[rand_indx, :][:, rand_indx]
    return features_batch, adj_label_batch


def random_pagerank(adj_label, idx_train, features, batch_size):
    # TODO @Fabi
    pass


def random_degree(adj_label, idx_train, features, batch_size):
    # TODO @Tobi
    pass


def rank_degree(adj_label, idx_train, features, batch_size):
    # TODO @Jan
    pass


def list_sampling(adj_label, idx_train, features, batch_size):
    # TODO @Jan
    pass


def negative_sampling(adj_label, idx_train, features, batch_size):
    # TODO @Fabi
    pass


def random_edge(adj_label, idx_train, features, batch_size):
    # TODO @Tobi
    pass


def random_node_edge(adj_label, idx_train, features, batch_size):
    # TODO @Tobi
    pass


def hybrid_edge(adj_label, idx_train, features, batch_size):
    # TODO @Tobi
    pass


def fixed_size_neighbor(adj_label, idx_train, features, batch_size):
    # TODO @Jan
    pass


def random_node_neighbor(adj_label, idx_train, features, batch_size):
    # TODO @Fabi
    pass


def random_walk(adj_label, idx_train, features, batch_size):
    # TODO @Jan
    pass


def random_jump(adj_label, idx_train, features, batch_size):
    # TODO @Jan
    pass


def forest_fire(adj_label, idx_train, features, batch_size):
    # TODO @Tobi
    pass


def frontier(adj_label, idx_train, features, batch_size):
    # TODO @Fabi
    pass


def snowball(adj_label, idx_train, features, batch_size):
    # TODO @Fabi
    pass
