#!/bin/bash

#SBATCH --job-name=parallel-ipynb
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=11
#SBATCH --time=4:00:00
#SBATCH --mem=16G

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen-jax

jupyter nbconvert --to notebook --execute --inplace notebooks/parallel.ipynb

rm slurm-$SLURM_JOBID.out
