export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_num=8
base_model_path="Mistral-7B-Instruct-v0.2"

vllm serve \
    $base_model_path \
    --served-model-name "HLS_model" \
    --port 8015 \
    --tensor-parallel-size $GPU_num \
    --dtype auto \
    --api-key "token-abc123" \
    --gpu_memory_utilization 0.9 \
    # --enable-prefix-caching
