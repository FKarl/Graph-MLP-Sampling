#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import homophily
from torch_geometric.datasets import Planetoid, Reddit, FacebookPagePage
import networkx as nx
from torch_geometric.utils import to_networkx


# 

# ## Calculate Cluster coefficient

# In[ ]:


def clustering_coef(data):
    # Convert data to a NetworkX graph
    G = to_networkx(data)

    # Compute the average cluster coefficient https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.average_clustering.html
    average_cc = nx.average_clustering(G)
    return average_cc


# ## Calculate hompholiy measures

# In[ ]:


def calc_homophily(data):
    print('Edge Homophily:', homophily(data.edge_index, data.y, method='edge'))
    print('Node Homophily:', homophily(data.edge_index, data.y, method='node'))
    print('Edge Insensitive Homophily:', homophily(data.edge_index, data.y, method='edge_insensitive'))


# # Dataset calculations:
# ## ogbn-arxiv:

# In[ ]:


# Download and process data at './dataset/ogbn_arxiv/'
dataset = PygNodePropPredDataset(name="ogbn-arxiv")

data = dataset[0]

print(data)
print('Undirected:', data.is_undirected())
calc_homophily(data)
print('Average Cluster Coefficient:', clustering_coef(data))


# ## ogbn-products:

# In[ ]:


# Download and process data at './dataset/ogbn_producs/'
dataset = PygNodePropPredDataset(name="ogbn-products")

data = dataset[0]

print(data)
print('Undirected:', data.is_undirected())
calc_homophily(data)
print('Average Cluster Coefficient:', clustering_coef(data))


# ## Citeseer:

# In[ ]:


# Download and process data at './dataset/Planetiod/Citeseer/'
dataset = Planetoid(root='dataset/Planetoid', name='Citeseer')

data = dataset[0]

print(data)
print('Undirected:', data.is_undirected())
calc_homophily(data)
print('Average Cluster Coefficient:', clustering_coef(data))


# ## Pubmed:

# In[ ]:


# Download and process data at './dataset/Planetiod/Pubmed/'
dataset = Planetoid(root='dataset/Planetoid', name='PubMed')

data = dataset[0]

print(data)
print('Undirected:', data.is_undirected())
calc_homophily(data)
print('Average Cluster Coefficient:', clustering_coef(data))


# ## Reddit:

# In[ ]:


# Download and process data at './dataset/Reddit/'
dataset = Reddit(root='dataset/Reddit')

data = dataset[0]

print(data)
print('Undirected:', data.is_undirected())
calc_homophily(data)
print('Average Cluster Coefficient:', clustering_coef(data))


# ## Facebook:

# In[ ]:


# Download and process data at './dataset/FacebookPagePage/'
dataset = FacebookPagePage(root='dataset/FacebookPagePage')

data = dataset[0]

print(data)
print('Undirected:', data.is_undirected())
calc_homophily(data)
print('Average Cluster Coefficient:', clustering_coef(data))


# # Cora

# In[ ]:


dataset = Planetoid(root='dataset/Planetoid', name='cora')

data = dataset[0]

print(data)
print('Undirected:', data.is_undirected())
calc_homophily(data)
print('Average Cluster Coefficient:', clustering_coef(data))

