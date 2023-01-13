#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=120:00
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias.kalmbach@uni-ulm.de

## cora
# python3 train.py --epochs=1000 --lr=0.001 --weight_decay=5e-3 --data=cora --alpha=10.0 --hidden=256 --batch_size=2000 --order=2 --tau=2

## citeseer
python3 train.py --epochs=1000 --lr=0.001 --weight_decay=5e-3 --data=citeseer --alpha=1.0 --hidden=256 --batch_size=2000 --order=2 --tau=0.5

## pubmed
python3 train.py --epochs=1000 --lr=0.1 --weight_decay=5e-3 --data=pubmed --alpha=100 --hidden=256 --batch_size=2000 --order=2 --tau=1

## ogbn-arxiv
python3 train.py --epochs=1000 --lr=0.1 --weight_decay=5e-3 --data=ogbn-arxiv --alpha=100 --hidden=256 --batch_size=2000 --order=2 --tau=1

## ogbn-products
python3 train.py --epochs=1000 --lr=0.1 --weight_decay=5e-3 --data=ogbn-products --alpha=100 --hidden=256 --batch_size=2000 --order=2 --tau=1

## Reddit
python3 train.py --epochs=1000 --lr=0.1 --weight_decay=5e-3 --data=reddit --alpha=100 --hidden=256 --batch_size=2000 --order=2 --tau=1

## FacebookPagePage
python3 train.py --epochs=1000 --lr=0.1 --weight_decay=5e-3 --data=facebook --alpha=100 --hidden=256 --batch_size=2000 --order=2 --tau=1