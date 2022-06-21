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