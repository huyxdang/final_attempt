# Quick Usage Guide

## Installation

```bash
pip install -r requirements.txt
```

## Complete Pipeline (Easiest)

Run everything at once:

```bash
# For MATH dataset (uses 2 GPUs, batch size 300 by default)
./pipeline.sh math train

# For MedQA dataset  
./pipeline.sh medqa train

# Override defaults if needed (e.g., use 4 GPUs)
./pipeline.sh math train 4
```

## Step-by-Step (More Control)

### 1. Run All Model Inferences

```bash
# Run all three models (uses defaults: 1 GPU for 3B, 2 GPUs for 72B, batch 300)
./run_all_models.sh math train
```

Arguments:
- Dataset: `math` or `medqa`
- Split: `train`, `test`, etc.
- Tensor parallel for 3B models (default: 1)
- Tensor parallel for 72B model (default: 2)

### 2. Create Difficulty Labels

```bash
python label_difficulty.py \
    --slm1_predictions predictions/qwen25_3b_instruct_math_train.json \
    --slm2_predictions predictions/ministral_3b_instruct_math_train.json \
    --llm_predictions predictions/qwen25_72b_math_train.json \
    --dataset_type math \
    --output labeled_datasets/math_train_labeled.json
```

## Individual Model Inference

Run one model at a time:

```bash
# Qwen 2.5 3B Instruct
python run_inference.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --dataset math \
    --split train \
    --output predictions/qwen3b_math_train.json

# Ministral 3B Instruct
python run_inference.py \
    --model_name "ministral/Ministral-3b-instruct" \
    --dataset math \
    --split train \
    --output predictions/ministral3b_math_train.json

# Qwen 2.5 72B (requires multiple GPUs)
python run_inference.py \
    --model_name "Qwen/Qwen2.5-72B" \
    --dataset math \
    --split train \
    --tensor_parallel_size 4 \
    --output predictions/qwen72b_math_train.json
```

## Output Files

After running the complete pipeline, you'll have:

```
predictions/
├── qwen25_3b_instruct_math_train.json
├── ministral_3b_instruct_math_train.json
└── qwen25_72b_math_train.json

labeled_datasets/
├── math_train_labeled.json                  # Full dataset with all labels
├── math_train_labeled_filtered.parquet      # Filtered (no label 2)
└── math_train_labeled_training.parquet      # Training-ready format
```

## Load Training Dataset

```python
from datasets import load_dataset

# Load the training-ready dataset
dataset = load_dataset(
    'parquet', 
    data_files='labeled_datasets/math_train_labeled_training.parquet'
)

# View first example
print(dataset['train'][0])
# {'question': '...', 'label': 0, 'correct_answer': '...'}

# Check label distribution
from collections import Counter
labels = [ex['label'] for ex in dataset['train']]
print(Counter(labels))
```

## GPU Requirements

| Model | Size | Recommended GPUs | VRAM |
|-------|------|------------------|------|
| Qwen 2.5 3B | 3B | 1 GPU | ~10GB |
| Ministral 3B | 3B | 1 GPU | ~10GB |
| Qwen 2.5 72B | 72B | 4-8 GPUs | ~320GB total |

For the 72B model, use tensor parallelism:
- 4 GPUs: `--tensor_parallel_size 4`
- 8 GPUs: `--tensor_parallel_size 8`

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 10`
- Increase tensor parallelism: `--tensor_parallel_size 8`

### Resuming Interrupted Runs
The scripts automatically resume. Just re-run the same command.

### Check Progress
Prediction files are saved incrementally. Check file sizes:
```bash
ls -lh predictions/
```

## Next Steps

1. **Analyze the labeled data**:
   ```python
   import json
   with open('labeled_datasets/math_train_labeled.json') as f:
       data = json.load(f)
   
   # Count labels
   from collections import Counter
   labels = [d['label'] for d in data]
   print(Counter(labels))
   ```

2. **Train a difficulty classifier**:
   - Use the `*_training.parquet` file
   - Fine-tune a small model (e.g., bert-base, distilbert)
   - Binary classification: easy (0) vs hard (1)

3. **Use the classifier** to filter training data for other models:
   - Train on "hard" examples for efficiency
   - Skip "easy" examples
   - Omit ambiguous examples

