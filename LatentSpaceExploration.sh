#!/bin/bash

#SBATCH --time 24:00:00
#SBATCH -p gpu --gres gpu:v100:4
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=karlberb@ohsu.edu,kirchgae@ohsu.edu
#SBATCH --mem 500G

source venv/bin/activate
python3 latent_space_exploration.py -cg $1 -ncg $2 -mf $3 -lt $4 -m $5