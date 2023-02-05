#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=120:00
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu_4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias.kalmbach@uni-ulm.de

python3 train.py --data=cora --epochs=400 --hidden=256 --dropout=0.6 --lr=0.001 --weight_decay=5e-3 --alpha=100.0 --batch_size=2000 --order=3 --tau=2 --sampler=random_batch
