#!/bin/bash

#SBATCH --time 1:00:00
#SBATCH -p gpu
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=karlberb@ohsu.edu,kirchgae@ohsu.edu

source venv/bin/activate
python3 vae.py -lt $1 -p $2 --train $3 --val $4 --test $5