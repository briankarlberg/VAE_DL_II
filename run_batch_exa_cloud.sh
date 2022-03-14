#!/bin/bash

for filename in ./data/*.tsv;
do
  sbatch ./VAE_exa_cloud_cmd_ln.sh $filename
done