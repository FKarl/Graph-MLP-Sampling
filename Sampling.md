# Sampling Strategies
This is a list of sampling strategies implemented in this extension.
For the `--sampler` argument use the name in parentheses.

### Random Batch Sampling (`random_batch`)
*N* randomly sampled nodes are selected and their corresponding adjacency information and node features are aggregated.

### Random PageRank Node (`random_pagerank`)
Non-uniform sampling, probability of a node being sampled is relative to its PageRank weight.

### Random Degree Node (`random_degree`)
Non-uniform sampling, the higher a nodes degree the higher the probability to be sampled; leads to very dense samples.

### Rank Degree (`rank_degree`)
Create a set *S* of the top *k* nodes by ranking the nodes according to their degree.
Rank the neighbors of a random node from *S*.
In order to create the new seed set *S* for the following iteration, the top *k* nodes of these iterations are added to the sub-graph.

### List Sampling (`list`)
Same as Rank Degree but in contrast we maintain a list of candidate nodes that is composed of all the neighbors of nodes that have been found but not yet sampled.
Experiments show that this method of sampling retains a better balance between the depth and breadth of the sampled sub-graph.

### Negative Sampling (`negative`)
Generates *n* source-destination pairs that do not share an edge.
It is possible that in small or very dense graphs less than *n* pairs are generated, because less than *n* pairwise-unconnected nodes in the graph exist.

### Random Edge (`random_edge`)
Edges are uniformly sampled instead of the nodes themselves.

### Random Node Edge (`random_node_edge`)
Select a starting node and select an outgoing edge from the starting node at random.
Add the connected node to the sample.
Repeat this procedure with a new node selected at random.
This method has less bias towards high degree nodes than *Random Edge*.

### Hybrid Edge (`hybrid_edge`)
Do one step of Random Node Edge with probability *p* (set to *0.8* in the paper) and a step of Random Edge otherwise.

### Fixed-Size Neighbor Sampling (`fixed_size_neighbor`)
Uniformly sample random nodes and add a fixed number *k* of its neighbor.
This keeps the computational expense constant for every batch.

### Random Node Neighbor(`random_node_neighbor`)
Select a random node and also add all its (outgoing) neighbors; might explode in dense graphs.

### Random Walk (`random_walk`)
Select a random node and 'walk' along randomly selected outgoing edges.
With probability of *0.15* (often used in literature) go back to first node and restart walk.
Can get stuck, but possible to use iteration counter at which to break and select a new start node.

### Random Jump (`random_jump`)
Same as Random Walk, but prevents getting stuck by jumping to a new random node with probability of *0.15* instead of only going back to start.

### Forest Fire (`forest_fire`)
Select a start node and start adding (burning) edges with corresponding nodes.
When an edge and its corresponding other node is burned, this node can also add its edges.

### Frontier Sampling (`frontier`)
By keeping track of a list of m nodes that represent the graph, M-dimensional dependent random walks can sample the data.
Choose a node at random from the list, take a random walk, and swap it out for the newly chosen node.
The edge which was chosen by the random walk is added to sample set.

### Snowball Expansion Sampling (`snowball`)
The algorithm makes an effort to select nodes from every community in a graph.
By maximizing an expansion factor, the algorithm aims to construct the sample in a greedy manner.
