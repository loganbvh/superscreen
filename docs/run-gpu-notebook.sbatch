#!/bin/bash

#SBATCH --job-name=gpu-ipynb
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=4:00:00
#SBATCH --mem=8G

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen-jax

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

jupyter nbconvert --to notebook --execute --inplace notebooks/gpu.ipynb
