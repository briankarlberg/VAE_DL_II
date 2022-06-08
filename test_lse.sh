#!/bin/bash

#SBATCH --time 1:00:00
#SBATCH -p gpu
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=karlberb@ohsu.edu,kirchgae@ohsu.edu
#SBATCH --mem 50G

source venv/bin/activate
python3 latent_space_exploration.py -cg $1 -ncg $2 -mf $3 -lt $4