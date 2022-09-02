#! /bin/bash
#SBATCH -p rtx2080
#SBATCH -N 1
#SBATCH --gres=gpu:0


python preprocess.py
