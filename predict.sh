#!/usr/bin/env bash

default_cuda_device=0
root_dir=/local/fever-common

# cleaning up
echo "cleaning up..."
rm /tmp/predicted_* 2> /dev/null
rm /tmp/rte_* 2> /dev/null

echo "--- batch inference ---"
python src/predict_batch.py --infile $1 --outfile $2
