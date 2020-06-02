#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

model_name=baseline
lr=3e-5
batch_size=16
epochs=15

save_model_freq=3
num_warmup_steps=500
threshold=0.5
nohup python -u run_zs.py \
--lr ${lr} \
--batch_size ${batch_size} \
--epochs ${epochs} \
--save_model_freq ${save_model_freq} \
--num_warmup_steps ${num_warmup_steps} \
--threshold ${threshold} \
--num_worker 8 \
> logs/train_${model_name}_lr_${lr}_bsz_${batch_size}.log \
2>&1 &





