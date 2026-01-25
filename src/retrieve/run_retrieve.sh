#!/bin/bash
#SBATCH --job-name=retrieve
#SBATCH --output=logs/retrieve_%A_%a.out
#SBATCH --error=logs/retrieve_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3  # 4 shards (0, 1, 2, 3)

# Usage:
#   sbatch run_retrieve.sh <input_csv> <output_jsonl> [num_shards]
#
# Example:
#   sbatch run_retrieve.sh data/input/contrastive_queries.csv data/output/contrastive_retrieved.jsonl
#   sbatch run_retrieve.sh data/input/prompts_placebo.csv data/output/placebo_retrieved.jsonl 8

# Parse arguments
INPUT=${1:?Usage: sbatch run_retrieve.sh <input_csv> <output_jsonl> [num_shards]}
OUTPUT=${2:?Usage: sbatch run_retrieve.sh <input_csv> <output_jsonl> [num_shards]}
NUM_SHARDS=${3:-4}

# Activate conda
source ~/.bashrc
conda activate medcorpus

# Set environment variables
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Print job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Shards: $SLURM_ARRAY_TASK_ID / $NUM_SHARDS"
echo "Started: $(date)"
echo "=========================================="

# Create output directory
mkdir -p logs
mkdir -p $(dirname $OUTPUT)

# Run retrieval for this shard
python -u retriever.py \
  --index ../../data/pyserini/index_full \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --stages full \
  --shard $SLURM_ARRAY_TASK_ID \
  --num-shards $NUM_SHARDS

echo "=========================================="
echo "Finished: $(date)"
echo "=========================================="