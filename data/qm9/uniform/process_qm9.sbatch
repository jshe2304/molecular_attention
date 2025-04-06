#!/bin/bash

#SBATCH --job-name=qm9
#SBATCH --error=qm9.err
#SBATCH --output=qm9.out
#SBATCH --account=pi-andrewferguson
#SBATCH --time=08:00:00
#SBATCH --partition=andrewferguson
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=8000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jshe@rcc.uchicago.edu

python3 processing.py