#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --time=120:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu_8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias.kalmbach@uni-ulm.de

DATASET="cora"
if [ "$1" != "" ]; then
    DATASET="$1"
fi

# model 'warm-up'
python3 train.py --data=cora --epochs=50 --hidden=256 --dropout=0.6 --lr=0.001 --weight_decay=5e-3 --alpha=100.0 --batch_size=200 --order=3 --tau=2 --sampler=random_batch --no-wandb

for samp in random_batch random_degree_higher random_degree_lower rank_degree negative random_edge random_node_edge hybrid_edge fixed_size_neighbor random_node_neighbor random_walk random_jump frontier
do
  for _ in {1..5}
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
      *)
        echo -n "Dataset not implemented."
        ;;
    esac
  done
done