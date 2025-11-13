# Difficulty Labeling Pipeline

Create difficulty-labeled datasets for training models to predict question difficulty.

## Overview

This pipeline runs inference on multiple language models and creates labeled datasets where:
- **Label 0 (Easy)**: At least one small language model (SLM) answers correctly
- **Label 1 (Hard)**: Both SLMs answer incorrectly, but the large language model (LLM) answers correctly
- **Label 2 (Omit)**: All models answer incorrectly

The goal is to create training data for fine-tuning a model to predict whether a question is easy, hard, or should be omitted.

## Models Used

1. **Small Language Models (SLMs)**:
   - [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
   - [Ministral 3B Instruct](https://huggingface.co/ministral/Ministral-3b-instruct)

2. **Large Language Model (LLM)**:
   - [Qwen 2.5 72B](https://huggingface.co/Qwen/Qwen2.5-72B)

## Supported Datasets

- **MATH**: Mathematical problem solving (`huyxdang/math-split`)
- **MedQA**: Medical question answering (`huyxdang/medqa-split`)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the complete pipeline with a single command:

```bash
./pipeline.sh math train
```

Arguments:
- `dataset`: `math` or `medqa` (default: `math`)
- `split`: Dataset split to use (default: `train`)
- `tensor_parallel_size`: Number of GPUs for tensor parallelism (default: `2`)
- `batch_size`: Batch size for inference (default: `300`)

### For MedQA:

```bash
./pipeline.sh medqa train
```

## Manual Pipeline Steps

If you prefer to run steps individually:

### Step 1: Run Inference on Qwen 2.5 3B Instruct

```bash
python run_inference.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --dataset math \
    --split train \
    --tensor_parallel_size 1 \
    --output predictions/qwen25_3b_instruct_math_train.json
```

### Step 2: Run Inference on Ministral 3B Instruct

```bash
python run_inference.py \
    --model_name "ministral/Ministral-3b-instruct" \
    --dataset math \
    --split train \
    --tensor_parallel_size 1 \
    --output predictions/ministral_3b_instruct_math_train.json
```

### Step 3: Run Inference on Qwen 2.5 72B

```bash
python run_inference.py \
    --model_name "Qwen/Qwen2.5-72B" \
    --dataset math \
    --split train \
    --tensor_parallel_size 4 \
    --output predictions/qwen25_72b_math_train.json
```

Note: The 72B model requires more GPU memory. Use `tensor_parallel_size` to distribute across multiple GPUs.

### Step 4: Create Difficulty Labels

```bash
python label_difficulty.py \
    --slm1_predictions predictions/qwen25_3b_instruct_math_train.json \
    --slm2_predictions predictions/ministral_3b_instruct_math_train.json \
    --llm_predictions predictions/qwen25_72b_math_train.json \
    --dataset_type math \
    --output labeled_datasets/math_train_labeled.json
```

## Output Files

The pipeline creates three types of output files:

1. **Full labeled dataset** (`*_labeled.json`):
   - Contains all examples with labels 0, 1, and 2
   - Includes model responses and correctness flags
   - Useful for analysis and debugging

2. **Filtered dataset** (`*_labeled_filtered.parquet`):
   - Excludes label 2 (omit) examples
   - Contains all metadata (responses, correctness, etc.)
   - HuggingFace Dataset format

3. **Training dataset** (`*_labeled_training.parquet`):
   - Only includes: question, label, correct_answer
   - Ready for fine-tuning
   - Minimal, clean format

## Data Format

### Training Dataset Schema

```python
{
    'question': str,      # The question text
    'label': int,         # 0 (easy), 1 (hard), or 2 (omit)
    'correct_answer': str # Ground truth answer
}
```

### Full Labeled Dataset Schema

```python
{
    'index': int,              # Original dataset index
    'question': str,           # The question text
    'correct_answer': str,     # Ground truth answer
    'label': int,              # 0 (easy), 1 (hard), 2 (omit)
    'slm1_correct': bool,      # Qwen 3B correctness
    'slm2_correct': bool,      # Ministral 3B correctness
    'llm_correct': bool,       # Qwen 72B correctness
    'slm1_response': str,      # Qwen 3B response
    'slm2_response': str,      # Ministral 3B response
    'llm_response': str        # Qwen 72B response
}
```

## Loading Labeled Datasets

### Python

```python
from datasets import load_dataset

# Load training dataset
dataset = load_dataset('parquet', data_files='labeled_datasets/math_train_labeled_training.parquet')

# Access examples
for example in dataset['train']:
    print(f"Question: {example['question']}")
    print(f"Label: {example['label']}")
    print(f"Answer: {example['correct_answer']}")
```

### With Pandas

```python
import pandas as pd

df = pd.read_parquet('labeled_datasets/math_train_labeled_training.parquet')
print(df.head())
print(df['label'].value_counts())
```

## Label Distribution Example

After running the pipeline, you'll see statistics like:

```
Label distribution:
  Easy (0): 450 (45.0%)
  Hard (1): 350 (35.0%)
  Omit (2): 200 (20.0%)
```

The filtered dataset excludes label 2, so you'll train only on easy and hard examples.

## GPU Requirements

- **3B Models**: 1 GPU with ~10GB VRAM
- **72B Model**: 4+ GPUs with tensor parallelism
  - Recommended: 4-8 A100 or H100 GPUs

## Next Steps

After generating labeled datasets:

1. **Fine-tune a difficulty classifier**:
   ```python
   from transformers import AutoModelForSequenceClassification, Trainer
   
   model = AutoModelForSequenceClassification.from_pretrained(
       "your-base-model",
       num_labels=2  # 0 (easy) and 1 (hard)
   )
   
   # Train with labeled_datasets/*_training.parquet
   ```

2. **Use the classifier** to filter training data:
   - Focus on "hard" examples for more efficient training
   - Skip "easy" examples that models already know
   - Omit examples that are too difficult or ambiguous

## Troubleshooting

### Out of Memory

If you run out of GPU memory:

```bash
# Reduce batch size
python run_inference.py --batch_size 10 ...

# Increase tensor parallelism (for 72B model)
python run_inference.py --tensor_parallel_size 8 ...
```

### Resuming Interrupted Runs

The inference script automatically resumes from where it left off. Just re-run the same command.

### Missing Predictions

Ensure all three prediction files exist before running `label_difficulty.py`. The script will show which files are missing.

## File Structure

```
final_attempt/
├── run_inference.py              # Run model inference
├── label_difficulty.py           # Create difficulty labels
├── pipeline.sh                   # Complete pipeline script
├── requirements.txt              # Python dependencies
├── predictions/                  # Model prediction outputs
│   ├── qwen25_3b_instruct_*.json
│   ├── ministral_3b_instruct_*.json
│   └── qwen25_72b_*.json
└── labeled_datasets/             # Final labeled datasets
    ├── *_labeled.json            # Full dataset with all metadata
    ├── *_labeled_filtered.parquet # Filtered (no omits)
    └── *_labeled_training.parquet # Training-ready format
```

## References

- [Qwen 2.5 3B Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Ministral 3B Instruct](https://huggingface.co/ministral/Ministral-3b-instruct)
- [Qwen 2.5 72B](https://huggingface.co/Qwen/Qwen2.5-72B)
- [vLLM Documentation](https://docs.vllm.ai/)
