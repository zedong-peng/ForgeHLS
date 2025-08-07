#!/bin/bash
set -e

# train
# dataset_name: PolyBench DB4HLS forgehls
# base_model: llama3-7B Qwen2.5-7B-Instruct
export CUDA_VISIBLE_DEVICES=0

dataset_path=./workspace/HLSBatchProcessor/downstream_task/pragma_insertion/dataset/pragma_insertion_forgehls_train_set.jsonl
base_model_path=./workspace/HLSBatchProcessor/downstream_task/QoR_prediction/models/models/Llama-3.2-1B-Instruct
output_dir=./workspace/HLSBatchProcessor/downstream_task/pragma_insertion/training_output/

# 确保输出目录存在
mkdir -p ${output_dir}

python train.py \
--dataset_path ${dataset_path} \
--base_model_path ${base_model_path} \
--output_dir ${output_dir} \
--num_epochs 2 \
--max_length 2048 \
--batch_size 2 \
--learning_rate 1e-5 