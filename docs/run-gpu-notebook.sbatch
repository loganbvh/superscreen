#!/bin/bash

#SBATCH --job-name=gpu-ipynb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --output=slurm-%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH --chdir=$HOME/superscreen/docs/notebooks

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen-jax

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

jupyter nbconvert --to notebook --execute --inplace gpu.ipynb
