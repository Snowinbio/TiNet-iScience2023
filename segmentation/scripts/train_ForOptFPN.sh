#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
#export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='thyroid.ForOptFPN'
model_dir='./log/thyroid/ForOptFPN'

python -m torch.distributed.launch --nproc_per_node=${NUM_GPUS} --master_port 9996 apex_train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    --opt_level='O1'
