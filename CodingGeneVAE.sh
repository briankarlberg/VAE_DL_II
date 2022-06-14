#!/bin/bash

#SBATCH --time 4:00:00
#SBATCH -p gpu --gres gpu:2
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=karlberb@ohsu.edu,kirchgae@ohsu.edu
#SBATCH --mem 100G

source venv/bin/activate
python3 coding_gene_vae.py -cg $1 -lt $2