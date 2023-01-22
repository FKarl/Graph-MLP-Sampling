# Graph-MLP Sampling Extension

> IMPORTANT: This is a fork of [Graph-MLP](https://github.com/yanghu819/Graph-MLP) in which we add new sampling strategies and other datasets.

PyTorch official implementation of *Graph-MLP: Node Classification without Message Passing in Graph*.
For details on the original Graph-MLP, please refer to the paper: https://arxiv.org/abs/2106.04051 

<img src="pipeline.png" width="60%" height="60%">

<img src="result.png" width="60%" height="60%">

## Requirements

  * PyTorch **1.7**
  * Python **3.7**

## Installation
First install PyTorch and PyTorch-Geometric (PyG).

Install the correct **PyTorch** version for your system from [the official website](https://pytorch.org/get-started/locally/).
Then install the corresponding **PyG** version (correct PyTorch version and same CUDA version) from [here](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

After this is done install the remaining requirements from the `requirements.txt` by running:
```shell
pip install -r requirements.txt
```

## Datasets
This is a list of datasets implemented in this extension. For the `--data` argument use the name in parentheses.
- [Cora](https://link.springer.com/article/10.1023/A:1009953814988) (`cora`), [CiteSeer](https://dl.acm.org/doi/10.1145/276675.276685) (`citeseer`), [Pubmed](https://ojs.aaai.org//index.php/aimagazine/article/view/2157) (`pubmed`): Citation networks, commonly used for a node classification baseline.
- [OGBN-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) (`ogbn-arxiv`): A large citation network based on papers from arxiv from 2017 to 2019.
- [OGBN-products](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) (`ogbn-products`): A co-purchasing network for Amazon products.
- [Reddit2](https://openreview.net/forum?id=BJe8pkHFwS) (`reddit2`): A condensed network of Reddit-posts connected by an overlapping commenter.
- [FacebookPagePage](https://academic.oup.com/comnet/article/9/2/cnab014/6271062) (`facebook`): A network of official Facebook pages connected by mutual likes.

> Note: OGBN-arxiv, OGBN-products and Reddit2 have many nodes. Make sure you have enough resources to run Graph-MLP on these datasets. See the table below for more information.

|          |  Cora | CiteSeer | Pubmed | ogbn-arxiv |    Reddit2 | Facebook |
|---------:|------:|---------:|-------:|-----------:|-----------:|---------:|
|    Nodes | 2,708 |    3,327 | 19,717 |    169,343 |    232,965 |   22,470 |
|    Edges | 5,429 |    4,732 | 44,338 |  1,166,243 | 23,213,838 |  171,002 |
|  Classes |     7 |        6 |      3 |         40 |         41 |        4 |
| Features | 1,433 |    3,703 |    500 |        128 |        602 |   14,000 |


## Usage

```
## cora
python3 train.py --lr=0.001 --weight_decay=5e-3 --data=cora --alpha=10.0 --hidden=256 --batch_size=2000 --order=2 --tau=2

## citeseer
python3 train.py --lr=0.001 --weight_decay=5e-3 --data=citeseer --alpha=1.0 --hidden=256 --batch_size=2000 --order=2 --tau=0.5

## pubmed
python3 train.py --lr=0.1 --weight_decay=5e-3 --data=pubmed --alpha=100 --hidden=256 --batch_size=2000 --order=2 --tau=1
```
or

```bash run.sh```

Please check our experimental results in log.txt. When new experiment is finished, the new result will also be appended to log.txt.

## Cite

As this repository only extends Graph-MLP, please still cite the original paper if you use this code in your own work:

```
@misc{hu2021graphmlp,
      title={Graph-MLP: Node Classification without Message Passing in Graph}, 
      author={Yang Hu and Haoxuan You and Zhecan Wang and Zhicheng Wang and Erjin Zhou and Yue Gao},
      year={2021},
      eprint={2106.04051},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
