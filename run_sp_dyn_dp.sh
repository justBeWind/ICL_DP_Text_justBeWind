#!/bin/bash
# run_sp_dyn_dp.sh
# Execute the SP-DYN-DP Framework (A + B + C)

echo "=========================================================="
echo " Starting SP-DYN-DP Processing (DSPP + DYNTEXT + DP-ST)   "
echo "=========================================================="
echo "Strict Professor Audit: Logging parameters out strictly..."

export CUDA_VISIBLE_DEVICES=0 # Update this based on the remote server GPU ID

python llama_dolly_noise.py \
    --eps 3.0 \
    --top_k 20 \
    --combine_method decode \
    --use_dynamic_k True \
    --use_structure True \
    --device cuda
