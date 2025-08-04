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

# Now run vLLM
CUDA_VISIBLE_DEVICES=2 vllm serve $LLM_NAME \
    --dtype auto \
    --api-key $VLLM_API_KEY \
    --port 6910 \
    --max-model-len 41440 &