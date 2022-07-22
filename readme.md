## Installation

1. Create virtual environment
2. Activate environment
   1. source venv/bin/activate
3. pip install -r requirements.txt
4. python3 src/main.py [arguments]

Command line args  (three head model)
1  coding genes  
2  non-coding genes  
3  morgan fingerprints  
4  scaling: min or s  
5  model: o or n   

# Usage

## VAE

# Current shell	file is	VAE.sh
# old .sh files	put to shell_script_archive

```
sbatch VAE.sh [path to file] [latent space size] [prefix]
```

Example:
```
sbatch VAE.sh data/coding_genes.ts 1000 coding_gene
```
Operations notes:
cd /home/groups/EllrottLab/drug_resp/VAE/VAE_DL_II/
scancel JOBID

2022-07-22

srun VAE-split.sh 1000 v0.4 X_exp_scal_xfrm_I-BET151_trn.tsv X_val_scal_xfrm_I-BET151_val.tsv X_exp_scal_xfrm_I-BET151_tst.tsv
sbatch VAE-split.sh 1000 v0.3 data/train_set.csv data/val_set.csv data/test_set.val

2022-07-21
sbatch VAE-split.sh 1000 v0.3 data/train_set.csv data/val_set.csv data/test_set.val

sbatch VAE.sh data/smp500_ftr5663.tsv 500 ncd_v0

2022-06-14
sbatch CodingGeneVAE.sh [path to file] [latent space size]
sbatch CodingGeneVAE.sh ./data/smp500_ftr14575.tsv 1000

$ srun -p gpu --gres gpu:1 pipeline.sh

# Usage

## VAE

```
sbatch VAE.sh [path to file] [latent space size] [prefix]
```

Example:
```
sbatch VAE.sh data/coding_genes.ts 1000 coding_gene
```