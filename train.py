from __future__ import division
from __future__ import print_function
import random
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import sample
from models import GMLP
from utils import accuracy, get_A_r, load_dataset
import warnings

import wandb

WANDB_ENTITY = "graph-mlp-sampling"
WANDB_PROJECT = "graph-mlp"

warnings.filterwarnings('ignore')

# Settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.6,
                    help='Dropout rate (1 - keep probability).')
# TODO adapt to all that are actually implemented and add description to README
parser.add_argument('--data', type=str,
                    choices=['cora', 'citeseer', 'pubmed', 'reddit2', 'ogbn-products', 'ogbn-arxiv', 'facebook'],
                    default='cora',
                    help="dataset to be used. Possible options are: 'cora', 'citeseer', 'pubmed', 'reddit2', "
                         "'ogbn-products', 'ogbn-arxiv', 'facebook'. See the README for more "
                         "information")
parser.add_argument('--alpha', type=float, default=2.0,
                    help='To control the ratio of Ncontrast loss')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='batch size')
parser.add_argument('--order', type=int, default=2,
                    help='to compute order-th power of adj')
parser.add_argument('--tau', type=float, default=1.0,
                    help='temperature for Ncontrast loss')
# TODO adapt to all that are actually implemented and add description to README
parser.add_argument('--sampler', type=str, choices=['random_batch', 'random_pagerank', 'random_degree', 'rank_degree',
                                                    'list', 'negative', 'random_edge', 'random_node_edge',
                                                    'hybrid_edge', 'fixed_size_neighbor', 'random_node_neighbor',
                                                    'random_walk', 'random_jump', 'forest_fire', 'frontier',
                                                    'snowball'],
                    default='random_batch',
                    help="sampler to use to generate a batch. Possible options are: 'random_batch', "
                         "'random_pagerank', 'random_degree', 'rank_degree', 'list', 'negative', 'random_edge', "
                         "'random_node_edge', 'hybrid_edge', 'fixed_size_neighbor', 'random_node_neighbor', "
                         "'random_walk', 'random_jump', 'forest_fire', 'frontier', 'snowball'. See the README for more "
                         "information")
parser.add_argument('--no-wandb', action='store_true', default=False,
                    help='Disables Weights & Biases logging.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Setup weights and biases
if not args.no_wandb:
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
               name=args.data + "-" + args.sampler + time.strftime("-%d%m%Y_%H%M%S", time.localtime()))
    wandb.config.update(args)

# get data
adj, features, labels, idx_train, idx_val, idx_test, edge_index = load_dataset(args.data, 'AugNormAdj', args.cuda)
print("DEBUG: Dataset loaded")
adj_label = get_A_r(adj, args.order)
print("DEBUG: Finished A_r calc")

# Model and optimizer
model = GMLP(nfeat=features.shape[1],
             nhid=args.hidden,
             nclass=labels.max().item() + 1,
             dropout=args.dropout,
             cuda=args.cuda
             )
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def Ncontrast(x_dis, adj_label, tau=1):
    """
    compute the Ncontrast loss
    """
    x_dis = torch.exp(tau * x_dis)
    x_dis_sum = torch.sum(x_dis, 1)
    x_dis_sum_pos = torch.sum(x_dis * adj_label, 1)
    loss = -torch.log(x_dis_sum_pos * (x_dis_sum ** (-1)) + 1e-8).mean()
    return loss


def train():
    features_batch, adj_label_batch, new_idx = sample.get_batch(adj_label, idx_train, features, edge_index,
                                                                batch_size=args.batch_size,
                                                                sampler=args.sampler, cuda=args.cuda)
    model.train()
    optimizer.zero_grad()
    output, x_dis = model(features_batch)
    loss_train_class = F.nll_loss(output[new_idx], labels[idx_train])
    loss_Ncontrast = Ncontrast(x_dis, adj_label_batch, tau=args.tau)
    loss_train = loss_train_class + loss_Ncontrast * args.alpha
    acc_train = accuracy(output[new_idx], labels[idx_train])
    if not args.no_wandb:
        wandb.log({"acc_train": acc_train, "loss_train_class": loss_train_class, "loss_Ncontrast": loss_Ncontrast,
                   "loss_train": loss_train})
    loss_train.backward()
    optimizer.step()
    return


def test():
    model.eval()
    output = model(features)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    if not args.no_wandb:
        wandb.log({"acc_test": acc_test, "loss_test": loss_test, "acc_val": acc_val})
    return acc_test, acc_val


best_accu = 0
best_val_acc = 0
print('\n' + 'training configs', args)
for epoch in tqdm(range(args.epochs)):
    train()
    tmp_test_acc, val_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc

log_file = open(r"log.txt", encoding="utf-8", mode="a+")
with log_file as file_to_be_write:
    print('tau', 'order',
          'batch_size', 'hidden',
          'alpha', 'lr',
          'weight_decay', 'data',
          'test_acc', file=file_to_be_write, sep=',')
    print(args.tau, args.order,
          args.batch_size, args.hidden,
          args.alpha, args.lr,
          args.weight_decay, args.data,
          test_acc.item(), file=file_to_be_write, sep=',')
