#!/bin/bash

for filename in ./data/*.tsv;
do
  sbatch ./VAE_exa_cloud_runner.sh $filename
done