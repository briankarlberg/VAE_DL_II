#!/bin/bash

#SBATCH --time 4:00:00
#SBATCH -p gpu --gres gpu:2
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=karlberb@ohsu.edu,kirchgae@ohsu.edu
#SBATCH --mem 100G

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/groups/EllrottLab/drug_resp/VAE/VAE_DL_II/venv/lib/python3.7/site-packages

export PATH=$PATH:/home/groups/EllrottLab/drug_resp/VAE/VAE_DL_II/graphviz_install/bin/bin

python3 latent_space_exploration.py -cg $1 -lt $2 -m cg