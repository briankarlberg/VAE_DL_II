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

