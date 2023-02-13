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
        if edge_index.numel() == 0:
            raise Exception('Only used training set as batch, but there are no edges in the training set. Raise the '
                            'batch_size above the number of training nodes (' + str(len(idx_train)) + ')')
        idx_train = torch.tensor(list(range(0, batch_size)))

    device = torch.device('cuda' if cuda else 'cpu')

    nodes = torch.unique(edge_index[0])
    # calculate and save the degree of all nodes
    if exists('data/degree_' + dataset + '.npy'):
        degrees = np.load('data/degree_' + dataset + '.npy')
    else:
        degrees = np.array([edge_index[0][edge_index[1] == node].shape[0] for node in nodes])
        np.save('data/degree_' + dataset + '.npy', degrees)

    if sampler == 'random_batch':
        return random_batch(adj_label, idx_train, features, labels, batch_size, device)
    elif sampler == 'random_degree_higher':
        return random_degree(edge_index, adj_label, idx_train, features, labels, batch_size, device, degrees,
                             higher_prob=True)
    elif sampler == 'random_degree_lower':
        return random_degree(edge_index, adj_label, idx_train, features, labels, batch_size, device, degrees,
                             higher_prob=False)
    elif sampler == 'rank_degree':
        return rank_degree(edge_index, adj_label, idx_train, features, labels, batch_size, device)
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
    elif sampler == 'frontier':
        return frontier(edge_index, adj_label, idx_train, features, labels, batch_size, device, degrees)
    elif sampler == 'snowball':
        return snowball(edge_index, adj_label, idx_train, features, labels, batch_size, device)


def idx_to_adj(node_index, idx_train, adj_label, features, labels, batch_size, device):
    node_index = node_index[:batch_size]
    if node_index.shape[0] < len(idx_train):
        new_idx = list(range(0, len(node_index)))
    elif len(idx_train) < batch_size:
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


def random_degree(edge_index, adj_label, idx_train, features, labels, batch_size, device, degrees, higher_prob=True):
    nodes = torch.unique(edge_index[0])
    total_degree = degrees.sum()
    if higher_prob:  # select nodes based on degree; higher degree ==> HIGHER selection probability
        selected_nodes = torch.tensor(
            np.random.choice(nodes.cpu(), batch_size, p=[deg / total_degree for deg in degrees])).type(torch.long).to(
            device)
    else:  # select nodes based on degree; higher degree ==> LOWER selection probability
        # calculate inverse degrees and sum them
        inverse_degree = [1 - deg / total_degree for deg in degrees]
        inverse_sum = sum(inverse_degree)
        selected_nodes = torch.tensor(
            np.random.choice(nodes.cpu(), batch_size, p=[deg / inverse_sum for deg in inverse_degree])).type(
            torch.long).to(
            device)

    return idx_to_adj(selected_nodes, idx_train, adj_label, features, labels, batch_size, device)


def rank_degree(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    s = 3  # number of randomly selected nodes as a starting point
    p = .35  # probability value defines the top-k of each ranking list
    seeds = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), s)).type(torch.long).to(device)
    sample = torch.Tensor(0).type(torch.long).to(device)

    while sample.shape[0] <= batch_size:
        new_seeds = torch.Tensor(0).type(torch.long).to(device)

        for w in seeds:
            neighbors = torch.tensor(edge_index[1, edge_index[0] == w]).type(torch.long).to(device)
            rank = torch.Tensor(neighbors.shape[0]).type(torch.long).to(device)

            # Calculates the degree of all neighbors, saves them in rank:
            for i in range(neighbors.shape[0]):
                rank[i] = torch.tensor(edge_index[0, edge_index[0] == neighbors[i]]).shape[0]

            # combine nodes with their rank degree (same format as edge_index)
            ranked_neighbors = torch.stack((neighbors, rank), 0).type(torch.long).to(device)

            # sort tensor based on the rank of each node highest to lowest degree:
            ranked_neighbors = ranked_neighbors[:, torch.argsort(ranked_neighbors[1, :], descending=True)]

            # select the k top ones
            k_top = ranked_neighbors[0][:int(ranked_neighbors[0].shape[0] * p)].type(torch.long).to(device)
            sample = torch.cat([sample, torch.tensor([w]).to(device), k_top])

            # add the other nodes as new_seeds
            new_seeds = torch.cat([new_seeds, k_top])

        seeds = torch.unique(new_seeds, sorted=False).type(torch.long).to(device)
        sample = torch.unique(sample, sorted=False).type(torch.long).to(device)

        # remove all nodes from the graph we already sampled:
        mask = ~torch.isin(edge_index[1, :], sample)
        edge_index = edge_index[:, mask]

        # if no seed has a degree >1 generate new random seeds:
        if not any(torch.tensor(edge_index[0, edge_index[0] == node.item()]).shape[0] > 1 for node in seeds):
            seeds = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), s)).type(torch.long).to(device)

    return idx_to_adj(sample, idx_train, adj_label, features, labels, batch_size, device)


def negative_sampling(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    # new edge index = all not existing edges
    new_edge_index = torch_geometric.utils.negative_sampling(edge_index)
    new_edge_index = new_edge_index.to(device)
    # select random batch_size  ((batch_size/2) edges from new_edge_index). So it will be always <= batch_size
    chosen_edges = torch.tensor(np.random.choice(np.arange(new_edge_index.shape[1]), int(batch_size / 2))).type(
        torch.long).to(device)
    chosen_nodes = torch.unique(new_edge_index[:, chosen_edges]).to(device)

    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)


def random_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device, from_hybrid=False):
    # select batch_size / 2 edges
    chosen_edges = torch.tensor(np.random.choice(np.arange(edge_index.shape[1]), int(batch_size / 2))).type(
        torch.long).to(device)
    # select and filter all nodes connected to the edges
    chosen_nodes = torch.unique(edge_index[:, chosen_edges]).to(device)
    # aggregate and return if sampler was used stand-alone, otherwise return only the nodes
    if not from_hybrid:
        return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)
    else:
        return chosen_nodes


def random_node_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device, from_hybrid=False):
    chosen_nodes = []
    # select batch_size / 2 nodes
    rand_indx = np.random.choice(np.arange(adj_label.shape[0]), int(batch_size / 2))
    for i in rand_indx:
        # for every node get connected neighbors ...
        connected_nodes = torch.tensor(edge_index[1, edge_index[0] == i]).type(torch.long).to(device)
        # ... and choose and add a neighbor to the sample
        new_node = connected_nodes[np.random.choice(np.arange(connected_nodes.shape[0]))]
        chosen_nodes.append(new_node)
    # convert to torch Tensor and filter duplicates
    chosen_nodes = torch.unique(
        torch.cat((torch.tensor(rand_indx), torch.tensor(chosen_nodes))).type(torch.long).to(device))
    # aggregate and return if sampler was used stand-alone, otherwise return only the nodes
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
    # Get nodes using both random_edge and random_node_edge; We do sample too many nodes (2 * batch_size) but this is
    # faster than sampling a single node batch_size times
    random_edges = random_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device, True)
    random_node_edges = random_node_edge(edge_index, adj_label, idx_train, features, labels, batch_size, device, True)

    # Select random nodes according to choices, or all nodes if there are less than chosen
    random_edges = random_edges[np.random.permutation(min(len(choices[choices == 1]), len(random_edges)))]
    random_node_edges = random_node_edges[
        np.random.permutation(min(len(choices[choices == 0]), len(random_node_edges)))]

    chosen_nodes = torch.unique(torch.cat((random_edges, random_node_edges))).type(torch.long).to(device)
    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)


def fixed_size_neighbor(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    FIXED_NC = 3  # max number of neighbors sampled per node
    K = 2  # how many layers are sampled

    # For the alternative:
    # max_spi =0 # max sample size per iteration
    # for i in range(K+1):
    #     max_spi += FIXED_NC**i
    # MAX_ITER = int(batch_size/max_spi)

    chosen_nodes = torch.empty(0).type(torch.long).to(device)
    # alternative: for k in range(MAX_ITER):
    while chosen_nodes.numel() < batch_size:
        start_node = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), 1)).type(torch.long).to(device)
        chosen_nodes = torch.concat([chosen_nodes, start_node], 0).type(torch.long).to(device)
        i = 0
        while i < K:
            i += 1
            for node in start_node:
                neighbors = edge_index[1, edge_index[0] == node].type(torch.long).to(device)
                # select fixed number of nodes, if there are not enough, select all neighbors:
                if not (neighbors.numel() < FIXED_NC):
                    neighbors = torch.tensor(np.random.choice(neighbors.cpu(), FIXED_NC, replace=False)).type(
                        torch.long).to(device)
                chosen_nodes = torch.concat([chosen_nodes, neighbors]).type(torch.long).to(device)
                start_node = neighbors

    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)


def random_node_neighbor(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    # we select a node uniformly at random together with all of its out-going neighbors.
    # empty tensor for new nodes
    chosen_nodes = torch.tensor([]).type(torch.long).to(device)
    edge_index = edge_index.to(device)
    while True:
        chosen_node = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), 1)).type(torch.long).to(device)
        outgoing_nodes = edge_index[1][edge_index[0] == chosen_node]
        # break if adding the new nodes would exceed the batch size
        if chosen_nodes.shape[0] + outgoing_nodes.shape[0] + 1 >= batch_size:
            break
        chosen_nodes = torch.cat((chosen_nodes, chosen_node, outgoing_nodes)).to(device)
    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)


def random_walk(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    # Jump back to start probability
    c = 0.15
    # select random node as starting point:
    start_node = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), 1)).type(torch.long).to(device)
    current_node = start_node.clone().detach()[0].to(device)
    sampled_nodes = start_node.clone().detach().to(device)
    neighbors = edge_index[1, edge_index[0] == current_node].to(device)

    # in case start node has no neighbors
    while neighbors.numel() == 0:
        start_node = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), 1)).type(torch.long).to(device)
        current_node = start_node.clone().detach()[0].to(device)
        sampled_nodes = start_node.clone().detach().to(device)
        neighbors = edge_index[1, edge_index[0] == current_node].to(device)

    for i in range(batch_size):

        neighbors = edge_index[1, edge_index[0] == current_node].to(device)

        # generate probability array for choosing the next node
        prob = np.ndarray((neighbors.numel() + 1))
        prob[:] = (1 - c) / (neighbors.numel())
        prob[0] = c

        # walk to one neighbor or the start_node
        merged_nodes = torch.concat([start_node, neighbors])
        current_node = merged_nodes[np.random.choice(merged_nodes.shape[0], p=prob)]

        # add the new current node to the sample
        if not (current_node in sampled_nodes):
            sampled_nodes = torch.concat([sampled_nodes, torch.tensor([current_node]).to(device)]).to(device)

    sampled_nodes = torch.tensor(sampled_nodes).type(torch.long).to(device)

    return idx_to_adj(sampled_nodes, idx_train, adj_label, features, labels, batch_size, device)


def random_jump(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    c = 0.15  # Probability to jump to a random node anywhere in the graph
    # select random node as starting point:
    random_node = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), 1)).type(torch.long).to(device)
    current_node = random_node.clone().detach()[0].to(device)
    sampled_nodes = random_node.clone().detach().to(device)

    while sampled_nodes.numel() < batch_size:
        neighbors = edge_index[1, edge_index[0] == current_node].to(device)

        if neighbors.numel() > 0:
            # generate probability array for choosing the next node
            prob = np.ndarray((neighbors.numel() + 1))
            prob[:] = (1 - c) / (neighbors.numel())
            prob[0] = c

            # walk to one neighbor or jump to random node
            random_node = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), 1)).type(torch.long).to(device)
            merged_nodes = torch.concat([random_node, neighbors])
            current_node = merged_nodes[np.random.choice(merged_nodes.shape[0], p=prob)]
        else:
            current_node = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), 1)).type(torch.long).to(device)
        # add the new current node to the sample
        if not (current_node in sampled_nodes):
            sampled_nodes = torch.concat([sampled_nodes, torch.tensor([current_node]).to(device)]).to(device)

    sampled_nodes = torch.tensor(sampled_nodes).type(torch.long).to(device)

    return idx_to_adj(sampled_nodes, idx_train, adj_label, features, labels, batch_size, device)


def frontier(edge_index, adj_label, idx_train, features, labels, batch_size, device, degrees):
    m = 100 if idx_train.shape[0] >= 100 else idx_train.shape[0]
    # init L with m randomly chosen nodes (uniformly)
    L = np.random.choice(np.arange(adj_label.shape[0]), m)
    chosen_nodes = torch.tensor(L).type(torch.long).to(device)
    # For to ensure maximum of batch_size nodes are chosen (duplicates are later removed)
    for iteration in range(batch_size):
        # calculate the degree of each node in L
        # current_degrees = np.array([edge_index[0][edge_index[1] == node].shape[0] for node in L])
        current_degrees = degrees[L]
        sum_of_degrees = current_degrees.sum()
        # if sum_of_degrees is 0 then all nodes in L have no neighbors, and we can't sample any more nodes
        if sum_of_degrees == 0:
            break
        # select random node u from L with probability degree(u)/sum_v in L degree(v)
        u = np.random.choice(L, p=[d / sum_of_degrees for d in current_degrees])
        # select random neighbor v of u
        outgoing_nodes = edge_index[1][edge_index[0] == u]
        # randomly choose one of the neighbors
        rand_idx = np.random.choice(np.arange(outgoing_nodes.shape[0]), 1)
        v = outgoing_nodes[rand_idx].cpu()
        # replace u with v in L
        L = np.where(L == u, v, L)
        # add v to chosen nodes
        chosen_nodes = torch.cat((chosen_nodes, torch.tensor([v]).type(torch.long).to(device)))
        # could be used if we want to sample exactly batch_size nodes (it's a bit slower, but not much)
        # chosen_nodes = torch.unique(torch.cat((chosen_nodes, torch.tensor([v]).type(torch.long).to(device))))
        # if chosen_nodes.shape[0] >= batch_size:
        #    break
    chosen_nodes = torch.unique(chosen_nodes)
    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)


def snowball(edge_index, adj_label, idx_train, features, labels, batch_size, device):
    v = torch.tensor(np.random.choice(np.arange(adj_label.shape[0]), 1)).type(torch.long).to(
        device)  # randomly chosen node
    chosen_nodes = v  # init chosen nodes with v

    # precalc N(x) for all nodes x that have been already computed
    # this is done to speed up the algorithm
    precaclulated_neighborhoods = torch.zeros((adj_label.shape[0], adj_label.shape[0]), dtype=torch.bool).to(device)

    # for instead of while to ensure maximum of batch_size nodes are chosen
    for iteration in range(batch_size):
        best_expansion_factor = -1
        best_v = -1
        # calc N(S)
        neighborhood_of_chosen_nodes = torch.unique(edge_index[1][torch.isin(edge_index[0], chosen_nodes)])
        ns_union_s = torch.unique(torch.cat((neighborhood_of_chosen_nodes, chosen_nodes)))
        # Select new node v ∈ N (S)
        for v in neighborhood_of_chosen_nodes:
            # calc neighborhood of v
            if precaclulated_neighborhoods[v].sum() == 0:
                neighborhood_of_v = torch.unique(edge_index[1][edge_index[0] == v])
                precaclulated_neighborhoods[v][neighborhood_of_v] = True
            else:
                neighborhood_of_v = precaclulated_neighborhoods[v].nonzero().squeeze()
            # remove nodes already in ns_union_s
            neighborhood_of_v = neighborhood_of_v[~torch.isin(neighborhood_of_v, ns_union_s)]
            # calc expansion factor
            expansion_factor = neighborhood_of_v.shape[0]

            if expansion_factor > best_expansion_factor:
                best_expansion_factor = expansion_factor
                best_v = v
        # add best node to chosen nodes
        chosen_nodes = torch.cat((chosen_nodes, torch.tensor([best_v]).type(torch.long).to(device)))

    return idx_to_adj(chosen_nodes, idx_train, adj_label, features, labels, batch_size, device)
