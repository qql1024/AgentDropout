# Load environment variables from .env
set -a
source .env
set +a

# Verify they're loaded
echo "API Key: '$VLLM_API_KEY'"
echo "Port: '$VLLM_PORT'"
echo "LLM_NAME: '$LLM_NAME'"

source /mnt/ccnas2/bdp/ql1024/miniconda3/bin/activate
conda activate agentdropout 

export HF_HOME=/mnt/ccnas2/bdp/ql1024/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# CUDA_VISIBLE_DEVICES=1 vllm serve "mistralai/Mistral-7B-Instruct-v0.3" \
#     --dtype auto \
#     --api-key $VLLM_API_KEY \
#     --port $VLLM_PORT \
#     --trust-remote-code \
#     --tokenizer-mode mistral &

# CUDA_VISIBLE_DEVICES=2 vllm serve "Qwen/Qwen2.5-7B-Instruct" \
#     --dtype auto \
#     --api-key $VLLM_API_KEY \
#     --port $(($VLLM_PORT + 1)) &

CUDA_VISIBLE_DEVICES=3 vllm serve "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --dtype auto \
    --api-key $VLLM_API_KEY \
    --port $(($VLLM_PORT + 5)) &

wait