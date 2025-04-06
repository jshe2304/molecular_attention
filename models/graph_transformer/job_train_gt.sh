#!/bin/bash

sbatch run.sbatch job_train_gt.py $1 $2 4LC
sbatch run.sbatch job_train_gt.py $1 $2 8C
sbatch run.sbatch job_train_gt.py $1 $2 8L
sbatch run.sbatch job_train_gt.py $1 $2 8M