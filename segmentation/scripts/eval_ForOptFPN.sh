#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:`pwd`

config_path='thyroid.ForOptFPN'
model_dir='./log/thyroid/ForOptFPN'
ckpt_path='./log/thyroid/ForOptFPN/model-60000.pth'
vis_dir='./log/thyroid/ForOptFPN/vis-60000'

image_dir='./thyroid/val/images'
mask_dir='./thyroid/val/masks'

python ./main_seg.py \
    --config_path=${config_path} \
    --ckpt_path=${ckpt_path} \
    --image_dir=${image_dir} \
    --mask_dir=${mask_dir} \
    --vis_dir=${vis_dir} \
    --log_dir=${model_dir} \
    --patch_size=896
