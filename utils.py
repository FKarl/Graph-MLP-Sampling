import numpy as np
import torch
from normalization import fetch_normalization, row_normalize

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid, Reddit2, FacebookPagePage
from torch_geometric.utils import mask_to_index, to_scipy_sparse_matrix
from torch_geometric.transforms import RandomNodeSplit


def get_A_r(adj, r):
    adj_label = adj.to_dense()
    if r == 1:
        adj_label = adj_label
    elif r == 2:
        adj_label = adj_label @ adj_label
    elif r == 3:
        adj_label = adj_label @ adj_label @ adj_label
    elif r == 4:
        adj_label = adj_label @ adj_label @ adj_label @ adj_label
    return adj_label


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def preprocess_dataset(adj, normalization="FirstOrderGCN", features=None):
    adj_normalizer = fetch_normalization(normalization)

    adj = adj_normalizer(adj)
    if features:
        features = row_normalize(features)
    return adj, features


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_dataset(dataset_str="cora", normalization="AugNormAdj", cuda=True):
    """
    Load All Datasets.
    """
    dataset_str = dataset_str.lower()
    if dataset_str in ['reddit2', 'ogbn-products', 'ogbn-arxiv'] and cuda:
        print("WARNING: The selected dataset is very large. It will probably not fit on a GPU. If you have an "
              "extremely powerful CPU and a lot of memory try adding --no-cuda.")
    if dataset_str in ['cora', 'citeseer', 'pubmed', 'reddit2']:
        dataset = None
        if dataset_str in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root='dataset/Planetoid', name=dataset_str)
        elif dataset_str in ['reddit2']:
            dataset = Reddit2(root='dataset/Reddit2')
        split = dataset.get(0)

        adj = to_scipy_sparse_matrix(split.edge_index).tocoo().astype(np.float32)
        features = split.x
        labels = split.y
        idx_train = mask_to_index(split.train_mask)
        idx_val = mask_to_index(split.val_mask)
        idx_test = mask_to_index(split.test_mask)

    elif dataset_str in ['ogbn-products', 'ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name=dataset_str)
        split = dataset.get(0)

        adj = to_scipy_sparse_matrix(split.edge_index).tocoo().astype(np.float32)
        features = split.x
        labels = split.y
        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]

    elif dataset_str in ['facebook']:
        dataset = FacebookPagePage(root='dataset/FacebookPagePage')
        split = dataset.get(0)
        transform = RandomNodeSplit(split='test_rest')
        transform(split)

        adj = to_scipy_sparse_matrix(split.edge_index).tocoo().astype(np.float32)
        features = split.x
        labels = split.y
        idx_train = mask_to_index(split.train_mask)
        idx_val = mask_to_index(split.val_mask)
        idx_test = mask_to_index(split.test_mask)

    else:
        raise Exception('Unknown dataset. The following datasets are supported: Cora, Citeseer, PubMed, '
                        'OGBN-Products, OGBN-Arxiv, Reddit2 and FacebookPagePage. For more information use the --help '
                        'option.')
    print('DEBUG: Finished creating dataset')
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    print('DEBUG: Finished creating dataset')

    adj, _ = preprocess_dataset(adj, normalization=normalization)
    print('DEBUG: Finished preprocessing dataset')
    # porting to pytorch
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print('DEBUG: Finished converting to pytorch')
    if cuda:
        features = features.cuda()
        if dataset_str not in ['reddit2', 'ogbn-products', 'ogbn-arxiv']:
            adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    return adj, features, labels, idx_train, idx_val, idx_test, split.edge_index
