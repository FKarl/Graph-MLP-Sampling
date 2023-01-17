#!/usr/bin/env python
# coding: utf-8

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import homophily
from torch_geometric.datasets import Planetoid, Reddit, FacebookPagePage
import networkx as nx
from torch_geometric.utils import to_networkx


# Calculate Cluster coefficient
def clustering_coef(data):
    # Convert data to a NetworkX graph
    G = to_networkx(data)

    # Compute the average cluster coefficient https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.average_clustering.html
    average_cc = nx.average_clustering(G)
    return average_cc


# Calculate hompholiy measures
def calc_homophily(data):
    print('Edge Homophily:', homophily(data.edge_index, data.y, method='edge'))
    print('Node Homophily:', homophily(data.edge_index, data.y, method='node'))
    print('Edge Insensitive Homophily:', homophily(data.edge_index, data.y, method='edge_insensitive'))


def calculate_stats(data):
    print(data)
    print('Undirected:', data.is_undirected())
    calc_homophily(data)
    print('Average Cluster Coefficient:', clustering_coef(data))


# Dataset calculations:
if __name__ == "__main__":
    # ogbn-arxiv:
    print("ogbn-arxiv")
    calculate_stats(PygNodePropPredDataset(name="ogbn-arxiv")[0])

    # ogbn-products:
    # print("ogbn-products")
    # calculate_stats(PygNodePropPredDataset(name="ogbn-products")[0])

    # Cora:
    print("Cora")
    calculate_stats(Planetoid(root='dataset/Planetoid', name='cora')[0])

    # Citeseer:
    print("Citeseer")
    calculate_stats(Planetoid(root='dataset/Planetoid', name='Citeseer')[0])

    # PubMed:
    print("PubMed")
    calculate_stats(Planetoid(root='dataset/Planetoid', name='PubMed')[0])

    # Reddit:
    print("Reddit")
    calculate_stats(Reddit(root='dataset/Reddit')[0])

    # Facebook:
    print("Facebook")
    calculate_stats(FacebookPagePage(root='dataset/FacebookPagePage')[0])
