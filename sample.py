from os.path import exists

import numpy as np
import torch
import torch_geometric


def get_batch(adj_label, idx_train, features, edge_index, labels, batch_size=2000, sampler='random_batch', cuda=True,
              dataset='cora'):
    # if batch_size is smaller than len(idx_train), remove everything except idx_train
    if batch_size < len(idx_train):
        if adj_label.device == torch.device('cpu'):
            cpu_idx_train = idx_train.cpu()
            adj_label = adj_label[cpu_idx_train, :][:, cpu_idx_train]
            idx_train = idx_train.to(torch.device('cuda' if cuda else 'cpu'))
        else:
            adj_label = adj_label[idx_train, :][:, idx_train]
        features = features[idx_train]
        labels = labels[idx_train]
        edge_index = edge_index[:, torch.isin(edge_index[0], idx_train) & torch.isin(edge_index[1], idx_train)]
        idx_train = torch.tensor(list(range(0, batch_size)))

    device = torch.device('cuda' if cuda else 'cpu')

    if sampler == 'random_batch':
        return random_batch(adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'random_pagerank':
        return random_pagerank(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'random_degree_higher':
        return random_degree(edge_index, adj_label, idx_train, features, labels, batch_size, device, dataset,
                             higher_prob=True)
    elif sampler == 'random_degree_lower':
        return random_degree(edge_index, adj_label, idx_train, features, labels, batch_size, device, dataset,
                             higher_prob=False)
    elif sampler == 'rank_degree':
        return rank_degree(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'list':
        return list_sampling(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'negative':
        return negative_sampling(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'random_edge':
        return random_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'random_node_edge':
        return random_node_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'hybrid_edge':
        return hybrid_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'fixed_size_neighbor':
        return fixed_size_neighbor(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'random_node_neighbor':
        return random_node_neighbor(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'random_walk':
        return random_walk(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'random_jump':
        return random_jump(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'forest_fire':
        return forest_fire(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'frontier':
        return frontier(edge_index, adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'snowball':
        return snowball(edge_index, adj_label, idx_train, features, labels, batch_size, device)


def idx_to_adj(node_index, idx_train, adj_label, features, labels, batch_size, device):
    node_index = node_index[:batch_size]
    if len(idx_train) < batch_size:
        node_index[0:len(idx_train)] = idx_train
        new_idx = list(range(0, len(idx_train)))
    else:
        new_idx = list(range(0, len(node_index)))
    features_batch = features[node_index]
    if adj_label.device == torch.device('cpu'):
        cpu_node_index = node_index.cpu()
        adj_label_batch = adj_label[cpu_node_index, :][:, cpu_node_index]
        node_index = node_index.to(device)
    else:
        adj_label_batch = adj_label[node_index, :][:, node_index]
    adj_label_batch = adj_label_batch.to(device)
    labels_batch = labels[node_index]
    return features_batch, adj_label_batch, labels_batch, new_idx


def random_batch(adj_label, idx_train, features, labels, batch_size, device):
    """
        get a batch of feature & adjacency matrix
    """
    rand_indx = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), batch_size)).type(torch.long).to(device)
    return idx_to_adj(rand_indx, idx_train, adj_label, features, labels, batch_size, device)


def random_pagerank(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    raise NotImplementedError('Not Implemented sampler!')


def random_degree(edge_index, adj_label, idx_train, features, labels, batch_size, device, dataset,
                  higher_prob=True):
    nodes = np.arange(adj_label.shape[0])
    # calculate and save the degree of all nodes
    if exists('degree_' + dataset + '.npy'):
        degrees = np.load('degree_' + dataset + '.npy')
    else:
        degrees = np.array([edge_index[0][edge_index[1] == node].shape[0] for node in nodes])
        np.save('degree_' + dataset + '.npy', degrees)
    total_degree = degrees.sum()
    if higher_prob:  # select nodes based on degree; higher degree ==> HIGHER selection probability
        selected_nodes = torch.tensor(
            np.random.choice(nodes, batch_size, p=[deg / total_degree for deg in degrees])).type(torch.long).to(device)
    else:  # select nodes based on degree; higher degree ==> LOWER selection probability
        # TODO how do I do that?
        selected_nodes = torch.tensor(
            np.random.choice(nodes, batch_size, p=[deg / total_degree for deg in degrees])).type(torch.long).to(device)
    return idx_to_adj(selected_nodes, idx_train, adj_label, features, labels, batch_size, device)


def rank_degree(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    s = 3  # number of randomly selected nodes as a starting point
    p = .35  # probability value defines the top-k of each ranking list

    seeds = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), s)).type(torch.long)
    sample = torch.Tensor(0)

    while sample.shape[0] <= batch_size:
        new_seeds = torch.Tensor(0)

        for w in seeds:
            neighbors = torch.tensor(edge_index[1, edge_index[0] == w])
            rank = torch.Tensor(neighbors.shape[0])

            # Calculates the degree of all neighbors, saves them in rank:
            for i in range(neighbors.shape[0]):
                rank[i] = torch.tensor(edge_index[0, edge_index[0] == neighbors[i]]).shape[0]

            # combine nodes with their rank degree (same format as edge_index)
            ranked_neighbors = torch.stack((neighbors, rank), 0)

            # sort tensor based on the rank of each node highes to lowest degree:
            ranked_neighbors = ranked_neighbors[:, torch.argsort(ranked_neighbors[1, :], descending=True)]

            # select the k top ones
            k_top = ranked_neighbors[0][:int(ranked_neighbors[0].shape[0] * p)]
            sample = torch.cat([sample, torch.tensor([w]), k_top])

            # add the other nodes as new_seeds
            new_seeds = torch.cat([new_seeds, k_top])

        seeds = torch.unique(new_seeds, sorted=False)
        sample = torch.unique(sample, sorted=False)

        # if no seed has a degree >1 generate new random seeds:
        if not any(torch.tensor(edge_index[0, edge_index[0] == node.item()]).shape[0] > 1 for node in seeds):
            seeds = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), s)).type(torch.long)

    # TODO: to keep sample size = batch_size remove the latest added nodes?
    sample = sample[:batch_size]

    return idx_to_adj(sample, idx_train, adj_label, features, labels, batch_size, device)


def list_sampling(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    raise NotImplementedError('Not Implemented sampler!')


def negative_sampling(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    # new edge index = all not existing edges
    new_edge_index = torch_geometric.utils.negative_sampling(edge_index)
    new_edge_index = new_edge_index.to(device)
    # select random batch_size edges
    chosen_edges = torch.tensor(np.random.choice(np.arange(new_edge_index.shape[1]), int(batch_size / 2))).type(
        torch.long).to(device)
    chosen_nodes = torch.unique(new_edge_index[:, chosen_edges]).to(device)

    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)


def random_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device, from_hybrid=False):
    chosen_edges = torch.tensor(np.random.choice(np.arange(edge_index.shape[1]), int(batch_size / 2))).type(
        torch.long).to(device)
    chosen_nodes = torch.unique(edge_index[:, chosen_edges]).to(device)
    if not from_hybrid:
        return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)
    else:
        return chosen_nodes


def random_node_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device, from_hybrid=False):
    chosen_nodes = []
    rand_indx = np.random.choice(np.arange(adj_label.shape[0]), int(batch_size / 2))
    for i in rand_indx:
        connected_nodes = torch.tensor(edge_index[1, edge_index[0] == i]).type(torch.long).to(device)
        new_node = connected_nodes[np.random.choice(np.arange(connected_nodes.shape[0]))]
        chosen_nodes.append(new_node)
    chosen_nodes = torch.unique(
        torch.cat((torch.tensor(rand_indx), torch.tensor(chosen_nodes))).type(torch.long).to(device))
    if not from_hybrid:
        return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)
    else:
        return chosen_nodes


def hybrid_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    random_node_edge_prob = 0.8
    # 0 is Random Node Edge; 1 is Random Edge
    choices = torch.tensor(
        np.random.choice(2, batch_size, p=[random_node_edge_prob, 1 - random_node_edge_prob])).type(
        torch.long).to(device)
    # Get nodes using both random_edge and random_node_edge
    random_edges = random_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device, True)
    random_node_edges = random_node_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device, True)

    # Select random nodes according to choices, or all nodes if there are less than chosen
    random_edges = random_edges[torch.randperm(min(len(choices[choices == 1]), len(random_edges)))]
    random_node_edges = random_node_edges[torch.randperm(min(len(choices[choices == 0]), len(random_node_edges)))]

    chosen_nodes = torch.unique(torch.cat((random_edges, random_node_edges))).type(torch.long).to(device)
    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)


def fixed_size_neighbor(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    # TODO @Jan
    pass


def random_node_neighbor(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    # we select a node uniformly at random together with all of its out-going neighbors.
    # empty tensor for new nodes
    chosen_nodes = torch.tensor([]).type(torch.long).to(device)
    edge_index = edge_index.to(device)
    while True:
        chosen_node = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), 1)).type(torch.long).to(device)
        outgoing_nodes = edge_index[1][edge_index[0] == chosen_node]
        chosen_nodes = torch.cat((chosen_nodes, chosen_node, outgoing_nodes)).to(device)
        if chosen_nodes.shape[0] >= batch_size:
            break
    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)


def random_walk(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    # if after max_steps the batch size is not filled change start node
    max_steps = batch_size * 100
    # Jump back to start probability
    c = 0.15
    # select random node as starting point:
    start_node = np.random.choice(np.arange(adj_label.shape[0]), 1)
    current_node = start_node
    sampled_nodes = start_node
    # in case start node has no neighbors
    while len(neighbors) == 0:
        start_node = np.random.choice(np.arange(adj_label.shape[0]), 1)
        current_node = start_node
        sampled_nodes = start_node
        neighbors = edge_index[1, edge_index[0] == current_node[0]].numpy()

    while sampled_nodes.size < batch_size:

        max_steps -= 1
        # in case batch_size cant be filled start over with a new start_node
        if (max_steps < 0):
            old_start = start_node
            while (old_start == start_node) or (len(neighbors) == 0):  # avoid same start and no neighbor
                start_node = np.random.choice(np.arange(adj_label.shape[0]), 1)
                neighbors = edge_index[1, edge_index[0] == start_node[0]].numpy()
            current_node = start_node
            sampled_nodes = start_node
            max_steps = (batch_size * 100) - 1

        neighbors = edge_index[1, edge_index[0] == current_node[0]].numpy()

        # generate probability array for choosing the next node
        prob = np.ndarray((len(neighbors) + 1))
        # TODO: Should be fine cause start node is checked for len(neighbors)=0
        prob[:] = (1 - c) / (len(neighbors))
        prob[0] = c
        # walk to one neighbor or the start_node
        current_node = np.array(
            [np.random.choice(np.concatenate([start_node, neighbors]), p=prob)])
        # add the new current node to the sample
        if not (current_node[0] in sampled_nodes):
            sampled_nodes = np.concatenate([sampled_nodes, current_node])

    sampled_nodes = torch.tensor(sampled_nodes).type(torch.long).to(device)

    return idx_to_adj(sampled_nodes, idx_train, adj_label, features, labels, batch_size, device)


def random_jump(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    c = 0.15  # Probability to jump to a random node anywher in the graph
    # select random node as starting point:
    random_node = np.random.choice(np.arange(adj_label.shape[0]), 1)
    current_node = random_node
    sampled_nodes = random_node

    while sampled_nodes.size < batch_size:

        neighbors = edge_index[1, edge_index[0] == current_node[0]].numpy()

        if len(neighbors) > 0:
            # generate probability array for choosing the next node
            prob = np.ndarray((len(neighbors) + 1))
            prob[:] = (1 - c) / (len(neighbors))
            prob[0] = c
            # walk to one neighbor or jump to random node
            random_node = np.random.choice(np.arange(adj_label.shape[0]), 1)
            current_node = np.array(
                [np.random.choice(np.concatenate([random_node, neighbors]), p=prob)])
        else:
            current_node = np.random.choice(np.arange(adj_label.shape[0]), 1)
        # add the new current node to the sample
        if not (current_node[0] in sampled_nodes):
            sampled_nodes = np.concatenate([sampled_nodes, current_node])

    sampled_nodes = torch.tensor(sampled_nodes).type(torch.long).to(device)

    return idx_to_adj(sampled_nodes, idx_train, adj_label, features, labels, batch_size, device)


def forest_fire(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    raise NotImplementedError('Not Implemented sampler!')


def frontier(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    # fixme @Fabi
    chosen_nodes = torch.tensor([]).type(torch.long).to(device)
    m = 10  # TODO tweak parameter and mention in section 3
    # init L with m randomly chosen nodes (uniformly)
    L = np.random.choice(np.arange(adj_label.shape[0]), m)
    while True:
        # calculate the degree of each node in L
        degrees = np.array([edge_index[0][edge_index[1] == node].shape[0] for node in L])
        sum_of_degrees = degrees.sum()
        # select randome node u from L with probability degree(u)/sum_v in L degree(v)
        u = np.random.choice(L, 1, p=[d / sum_of_degrees for d in degrees])
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


def snowball(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    # TODO @Fabi
    pass
