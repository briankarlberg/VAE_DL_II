#!/bin/bash

#SBATCH --time 24:00:00
#SBATCH -p gpu --gres gpu:v100:1
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=karlberb@ohsu.edu,kirchgae@ohsu.edu
#SBATCH --mem 128G

source venv/bin/activate
python3 grid_search_hyper_parameter_tuning.py -cg $1 -ngc $2 -mf $3