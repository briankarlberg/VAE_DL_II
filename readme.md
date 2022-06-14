## Installation

1. Create virtual environment
2. Activate environment
   1. source venv/bin/activate
3. pip install -r requirements.txt
4. python3 src/main.py [arguments]

Command line args  
1  coding genes  
2  non-coding genes  
3  morgan fingerprints  
4  scaling: min or s  
5  model: o or n   

Operations notes:
cd /home/groups/EllrottLab/drug_resp/VAE/VAE_DL_II/
scancel JOBID

2022-06-14
sbatch CodingGeneVAE.sh ./data/smp500_ftr19154_dcm3.tsv 1000

sbatch CodingGeneVAE.sh ./data/coding_1_2_decimals.tsv 1000

Local run:
python3 coding_gene_vae.py -cg ./data/coding_1_2_decimals.tsv -lt 1000
python3 coding_gene_vae.py -cg ./data/cod_round3_samp500.tsv -lt 1000

File with rounded decimal:
sbatch CodingGeneVAE.sh ./data/cod_round3_samp500.tsv 1000

File with long decimals
sbatch CodingGeneVAE.sh data/500_samples_v1/coding_1.tsv 1000

2022-06-10
import pydot_ng as pydot
pydot.Dot.create(pydot.Dot())

coding_gene_vae.py

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/users/karlberb/gsl/lib
/home/groups/EllrottLab/drug_resp/VAE/VAE_DL_II/venv/lib/python3.7/site-packages

$ srun -p gpu --gres gpu:1 pipeline.sh

2022-06-09
sbatch dev_CGLatentSpaceExploration.sh data/500_samples_v1/coding_1.tsv 1000 o

2022-06-08
sbatch lse_500.sh data/500_samples_v1/coding_1.tsv data/500_samples_v1/noncod_1.tsv data/500_samples_v1/morgan_1.tsv 1000

sbatch test_lse.sh data/5_samples/coding_5.tsv data/5_samples/noncod_5.tsv data/5_samples/morgan_5.tsv 1000

# Usage

## Coding Gene VAE

```
sbatch CodingGeneVAE.sh [path to file] [latent space size]
```

## Non Coding Gene VAE

TBD

## Molecular Fingerprint VAE

TBD

