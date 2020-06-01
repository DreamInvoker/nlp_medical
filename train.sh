#! /bin/bash
export CUDA_VISIBLE_DEVICE=$1
lr=3e-5
batch_size=16
epochs=1000
model_name=baseline
save_model_freq=3
nohup python -u train_zs.py \
--lr ${lr} \
--batch_size ${batch_size} \
--epochs ${epochs} \
--log_step 50 \
--save_model_freq ${save_model_freq} \
>> logs/train_${model_name}.log \
2>&1 &





