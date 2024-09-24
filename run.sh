#!/bin/bash

#SBATCH --job-name=RL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --partition=GPU
#SBATCH --output=out.out
#SBATCH --error=err.err
#SBATCH --mem=32GB

module load cuda
source ../rlenv/bin/activate
python3 main.py
