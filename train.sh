#! /bin/bash
export CUDA_VISIBLE_DEVICE=$1
batch_size=16
epochs=1000
model_name=baseline

nohup python -u train_zs.py \
--batch_size ${batch_size} \
--epochs ${epochs} \
--log_step 50 \
>> logs/train_${model_name}.log \
2>&1 &





