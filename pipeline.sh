#!/bin/bash

# Complete pipeline for creating difficulty-labeled datasets
# 
# Usage: ./pipeline.sh [dataset] [split] [tensor_parallel_size]
#
# Example: ./pipeline.sh math train 2

set -e  # Exit on error

DATASET=${1:-math}
SPLIT=${2:-train}
TENSOR_PARALLEL=${3:-2}
BATCH_SIZE=${4:-300}

echo "============================================"
echo "Difficulty Labeling Pipeline"
echo "============================================"
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Tensor Parallel Size: $TENSOR_PARALLEL"
echo "Batch Size: $BATCH_SIZE"
echo "============================================"
echo ""

# Create output directories
mkdir -p predictions
mkdir -p labeled_datasets

# Step 1: Run inference with Qwen 2.5 3B Instruct
echo "============================================"
echo "STEP 1/4: Running Qwen 2.5 3B Instruct"
echo "============================================"
python run_inference.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --tensor_parallel_size "$TENSOR_PARALLEL" \
    --batch_size "$BATCH_SIZE" \
    --output "predictions/qwen25_3b_instruct_${DATASET}_${SPLIT}.json"

echo ""

# Step 2: Run inference with Ministral 3B Instruct
echo "============================================"
echo "STEP 2/4: Running Ministral 3B Instruct"
echo "============================================"
python run_inference.py \
    --model_name "ministral/Ministral-3b-instruct" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --tensor_parallel_size "$TENSOR_PARALLEL" \
    --batch_size "$BATCH_SIZE" \
    --output "predictions/ministral_3b_instruct_${DATASET}_${SPLIT}.json"

echo ""

# Step 3: Run inference with Qwen 2.5 72B
echo "============================================"
echo "STEP 3/4: Running Qwen 2.5 72B"
echo "============================================"
python run_inference.py \
    --model_name "Qwen/Qwen2.5-72B" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --tensor_parallel_size "$TENSOR_PARALLEL" \
    --batch_size "$BATCH_SIZE" \
    --output "predictions/qwen25_72b_${DATASET}_${SPLIT}.json"

echo ""

# Step 4: Create difficulty labels
echo "============================================"
echo "STEP 4/4: Creating Difficulty Labels"
echo "============================================"
python label_difficulty.py \
    --slm1_predictions "predictions/qwen25_3b_instruct_${DATASET}_${SPLIT}.json" \
    --slm2_predictions "predictions/ministral_3b_instruct_${DATASET}_${SPLIT}.json" \
    --llm_predictions "predictions/qwen25_72b_${DATASET}_${SPLIT}.json" \
    --dataset_type "$DATASET" \
    --output "labeled_datasets/${DATASET}_${SPLIT}_labeled.json"

echo ""
echo "============================================"
echo "PIPELINE COMPLETE!"
echo "============================================"
echo ""
echo "Output files:"
echo "  - Full labeled dataset: labeled_datasets/${DATASET}_${SPLIT}_labeled.json"
echo "  - Filtered dataset (no omits): labeled_datasets/${DATASET}_${SPLIT}_labeled_filtered.parquet"
echo "  - Training dataset: labeled_datasets/${DATASET}_${SPLIT}_labeled_training.parquet"
echo ""
echo "Next steps:"
echo "  Use the training dataset to fine-tune a model to predict difficulty"
echo ""

