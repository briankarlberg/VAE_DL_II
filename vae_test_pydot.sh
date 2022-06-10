#!/bin/bash

#SBATCH --time 1:00:00
#SBATCH -p gpu --gres gpu:1
#SBATCH --output=./output_reports/slurm.%N.%j.out
#SBATCH --error=./error_reports/slurm.%N.%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=karlberb@ohsu.edu,kirchgae@ohsu.edu
#SBATCH --mem 10G

source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/groups/EllrottLab/drug_resp/VAE/VAE_DL_II/venv/lib/python3.7/site-packages
export PATH=/home/groups/EllrottLab/drug_resp/VAE/VAE_DL_II/venv/bin:/home/groups/EllrottLab/gdan-tmp/bk_sandbox/toolbox/mini_conda/condabin:/home/users/karlberb/miniconda3/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/ibutils/bin:/opt/puppetlabs/bin:/opt/dell/srvadmin/bin:/home/users/karlberb/bin

python3 vae_test_pydot.py