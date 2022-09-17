#! /bin/bash
#SBATCH -p rtx2080
#SBATCH -N 1
#SBATCH --gres=gpu:2

python -m torch.distributed.launch  --nproc_per_node=2 main.py
