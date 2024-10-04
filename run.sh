#!/bin/bash

#SBATCH --job-name=RL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --partition=GPU
#SBATCH --output=out.txt
#SBATCH --error=err.err
#SBATCH --mem=32GB

module load cuda
source ../myenv/bin/activate
python3 ac_train.py --cluster
