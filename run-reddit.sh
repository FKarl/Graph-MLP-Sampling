#!/bin/bash
#SBATCH --ntasks=8
#SBATCH --time=30:00:00
#SBATCH --mem=500000
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu_8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias.kalmbach@uni-ulm.de


SAMPLER="random_batch"
if [ "$1" != "" ]; then
    SAMPLER="$1"
fi

python3 train.py --data=reddit2 --epochs=400 --hidden=2048 --dropout=0.15 --lr=0.001 --weight_decay=0 --alpha=30.0 --batch_size=7000 --order=3 --tau=15 --sampler="$SAMPLER"
