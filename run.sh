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
python3 ac_mp.py --cluster --num_workers=4 --stage 2 --save_file data/ac2.csv --load_param weights/a3c_super_mario_bros_1_2
