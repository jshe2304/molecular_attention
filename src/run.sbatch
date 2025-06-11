#!/bin/bash

#SBATCH --job-name=graph_transformer
#SBATCH --error=run.err
#SBATCH --output=/dev/null
#SBATCH --account=pi-andrewferguson
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --array=1-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

python3 train.py $1 $2 $3