#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=120:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu_4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias.kalmbach@uni-ulm.de

DATASET="cora"
if [ "$1" != "" ]; then
    DATASET="$1"
fi

for samp in random_batch random_degree_higher random_degree_lower rank_degree negative random_edge random_node_edge hybrid_edge fixed_size_neighbor random_node_neighbor random_walk random_jump frontier snowball # random_pagerank forest_fire list
do
  case $DATASET in
    cora) ## cora
      python3 train.py --data=cora --epochs=400 --hidden=256 --dropout=0.6 --lr=0.001 --weight_decay=5e-3 --alpha=100.0 --batch_size=2000 --order=3 --tau=2 --sampler=$samp
      ;;
    citeseer) ## citeseer
      python3 train.py --data=citeseer --epochs=400 --hidden=256 --dropout=0.6 --lr=0.01 --weight_decay=5e-3 --alpha=1.0 --batch_size=2000 --order=2 --tau=1 --sampler=$samp
      ;;
    pubmed) ## pubmed
      python3 train.py --data=pubmed --epochs=400 --hidden=256 --dropout=0.6 --lr=0.001 --weight_decay=5e-3 --alpha=1.0 --batch_size=3000 --order=3 --tau=2 --sampler=$samp
      ;;
    facebook) ## FacebookPagePage
      python3 train.py --data=facebook --epochs=400 --hidden=256 --dropout=0.6 --lr=0.001 --weight_decay=5e-4 --alpha=1.0 --batch_size=2000 --order=4 --tau=0.5 --sampler=$samp
      ;;
    ogbn-arxiv) ## ogbn-arxiv
      python3 train.py --data=ogbn-arxiv --epochs=400 --hidden=2048 --dropout=0.15 --lr=0.001 --weight_decay=0 --alpha=30.0 --batch_size=7000 --order=3 --tau=15 --sampler=$samp
      ;;
    reddit2) ## Reddit2
      python3 train.py --data=reddit2 --epochs=400 --hidden=2048 --dropout=0.15 --lr=0.001 --weight_decay=0 --alpha=30.0 --batch_size=7000 --order=3 --tau=15 --sampler=$samp
      ;;
    ogbn-products) ## ogbn-products
      python3 train.py --data=ogbn-products --epochs=400 --hidden=2048 --dropout=0.15 --lr=0.001 --weight_decay=0 --alpha=30.0 --batch_size=7000 --order=3 --tau=15 --sampler=$samp
      ;;
    *)
      echo -n "Dataset not implemented."
      ;;
  esac
done
