#!/bin/bash
#SBATCH --job-name=train_mk
#SBATCH --partition=all
#SBATCH --nodes=2
#SBATCH --time=20:00:00
#SBATCH --output slurm.%J.ou
#SBATCH --error slurm.%J.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vlajkojjj@gmail.com



source env/bin/activate

python3 -m retro.import ~/projects/ProjectForComputerVision/

python3 train_model.py
