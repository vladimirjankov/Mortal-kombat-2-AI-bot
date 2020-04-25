#!/bin/bash
#SBATCH --job-name=test__mk
#SBATCH --partition=cuda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --output slurm.%J.ou
#SBATCH --error slurm.%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vlajkojjj@gmail.com

python3 -m retro.import ~/projects/ProjectForComputerVision/

python3 test_model.py

