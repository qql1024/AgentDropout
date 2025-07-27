#!/bin/bash
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mail-type=ALL                # Email notification for all job events
#SBATCH --mail-user=ql2024    # Replace <your_username> with your actual username or email
#SBATCH --job-name=mas_benchmark   # Set the name of the job
#SBATCH --time=96:00:00                # Set the maximum time limit for the job (adjust as needed)
#SBATCH --mem=16GB                    # Request memory (adjust if needed)
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --output=/mnt/ccnas2/bdp/ql1024/X-MAS/outputs/err_and_out/slurm_%j.out     # <<<< Changed path for stdout
#SBATCH --error=/mnt/ccnas2/bdp/ql1024/X-MAS/outputs/err_and_out/slurm_%j.err         # Standard error will go to slurm_JOBID.err

# Print job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"

# Activate the python environment
source /mnt/ccnas2/bdp/ql1024/miniconda3/bin/activate
conda activate agentdropout  # Adjust this to the location of your virtualenv

# Set CUDA environment variable for debugging
# export CUDA_LAUNCH_BLOCKING=1  # Enable synchronous CUDA errors for debugging

# export CUDA_VISIBLE_DEVICES=1
# echo "Running LLM inference on GPUs: $CUDA_VISIBLE_DEVICES"
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" # Add this line

/usr/bin/nvidia-smi

echo "Available GPUs:"
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv

# export HF_HOME=/mnt/ccnas2/bdp/ql1024/huggingface
# export HF_HOME=/home/ql1024/huggingface

# export TRANSFORMERS_CACHE=$HF_HOME

set -a
source .env
set +a
echo "LLM_NAME: '$LLM_NAME'"


# Line 38 (or the relevant line):
python experiments/run_multiarith.py \
  --agent_nums 5 \
  --mode FullConnected \
  --batch_size 40 \
  --num_iterations 2 \
  --imp_per_iterations 1 \
  --pruning_rate 0.10 \
  --num_rounds 2 \
  --llm_name $LLM_NAME \
  --optimized_spatial \
  --optimized_temporal \
  --diff \
  --dec