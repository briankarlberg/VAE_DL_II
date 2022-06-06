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
scancel JOBID

2022-06-06

python3 latent_space_exploration.py -cg $1 -ncg $2 -mf $3 -s $4 -m $5

sbatch LSE.sh -cg data/coding_5.tsv -ncg data/noncod_5.tsv -mf data/morgan_5.tsv -lt 1000

[karlberb@exahead1 error_reports]$ cat slurm.exanode-8-5.19769070.err
2022-06-06 15:56:53.903962: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/karlberb/lib:
2022-06-06 15:56:53.908060: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
usage: latent_space_exploration.py [-h] -cg CODING_GENES -ncg NON_CODING_GENES
                                   -mf MOLECULAR_FINGERPRINT
                                   [-r RESPONSE_DATA] -lt LATENT_SPACE
                                   [-s {min,s}] [-m {o,n,r}]
latent_space_exploration.py: error: argument -s/--scaling: expected one argument
gpu:2
#SBATCH --gres=gpu:1

Generic gpu call on 5 sample files, 2022-05-25
sbatch LSE.sh data/coding_5.tsv data/noncod_5.tsv data/morgan_5.tsv min o

For 5 sample files, test on 2022-05-25
sbatch LatentSpaceExploration.sh data/coding_5.tsv data/noncod_5.tsv data/morgan_5.tsv min o

For 500 sample files, moved to 500_sample sub dir in data
sbatch LatentSpaceExploration.sh data/coding_1.tsv data/noncod_1.tsv data/morgan_1.tsv min o

Application development notes:

standard scaler does not use feature range

