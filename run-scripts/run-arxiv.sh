#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --time=40:00:00
#SBATCH --mem=350000
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu_8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias.kalmbach@uni-ulm.de

SAMPLER="random_batch"
if [ "$1" != "" ]; then
    SAMPLER="$1"
fi

for _ in {1..3}
do

  python3 train.py --data=ogbn-arxiv --epochs=400 --hidden=2048 --dropout=0.15 --lr=0.001 --weight_decay=0 --alpha=30.0 --batch_size=7000 --order=3 --tau=15 --sampler="$SAMPLER"

done
