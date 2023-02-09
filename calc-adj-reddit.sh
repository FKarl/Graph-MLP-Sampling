#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=10
#SBATCH --time=36:00:00
#SBATCH --mem=512000
#SBATCH --partition=fat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tobias.kalmbach@uni-ulm.de


python3 train.py --data=reddit2 --epochs=1 --hidden=2048 --dropout=0.15 --lr=0.001 --weight_decay=0 --alpha=30.0 --batch_size=7000 --order=3 --tau=15
