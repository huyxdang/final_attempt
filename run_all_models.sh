#!/bin/bash

# Run inference for all three models on a specific dataset
# 
# Usage: ./run_all_models.sh [dataset] [split] [tensor_parallel_3b] [tensor_parallel_72b]
#
# Example: ./run_all_models.sh math train 1 4

set -e

DATASET=${1:-math}
SPLIT=${2:-train}
TP_3B=${3:-1}
TP_72B=${4:-2}
BATCH_SIZE=${5:-300}

echo "============================================"
echo "Running All Models for Inference"
echo "============================================"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "3B Tensor Parallel: $TP_3B"
echo "72B Tensor Parallel: $TP_72B"
echo "Batch Size: $BATCH_SIZE"
echo "============================================"
echo ""

mkdir -p predictions

# Run Qwen 2.5 3B Instruct
echo "============================================"
echo "Running Qwen 2.5 3B Instruct"
echo "============================================"
python run_inference.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --tensor_parallel_size "$TP_3B" \
    --batch_size "$BATCH_SIZE" \
    --output "predictions/qwen25_3b_instruct_${DATASET}_${SPLIT}.json"

echo ""

# Run Ministral 3B Instruct
echo "============================================"
echo "Running Ministral 3B Instruct"
echo "============================================"
python run_inference.py \
    --model_name "ministral/Ministral-3b-instruct" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --tensor_parallel_size "$TP_3B" \
    --batch_size "$BATCH_SIZE" \
    --output "predictions/ministral_3b_instruct_${DATASET}_${SPLIT}.json"

echo ""

# Run Qwen 2.5 72B
echo "============================================"
echo "Running Qwen 2.5 72B"
echo "============================================"
python run_inference.py \
    --model_name "Qwen/Qwen2.5-72B" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --tensor_parallel_size "$TP_72B" \
    --batch_size "$BATCH_SIZE" \
    --output "predictions/qwen25_72b_${DATASET}_${SPLIT}.json"

echo ""
echo "============================================"
echo "All Models Complete!"
echo "============================================"
echo ""
echo "Prediction files:"
echo "  - predictions/qwen25_3b_instruct_${DATASET}_${SPLIT}.json"
echo "  - predictions/ministral_3b_instruct_${DATASET}_${SPLIT}.json"
echo "  - predictions/qwen25_72b_${DATASET}_${SPLIT}.json"
echo ""
echo "Next: Run label_difficulty.py to create labeled dataset"
echo ""

