#! /bin/bash
batch_size=16
epochs=100
model_name=baseline
export CUDA_VISIBLE_DEVICE=$1
nohup python -u train_zs.py \
--batch_size ${batch_size} \
--epochs ${epochs} \
>> logs/train_${model_name}.log \
2>&1 &





