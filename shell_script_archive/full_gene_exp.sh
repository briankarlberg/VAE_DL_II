#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name=vae_exa_cloud.sh
#SBATCH --time=0-24:00:00
#SBATCH --partition=exacloud
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=karlberb@ohsu.edu

source venv/bin/activate
python3 ./full_gene_exp.py -di "$1" -cl "$2" -dr "$3"