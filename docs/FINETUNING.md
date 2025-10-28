# Fine-tuning Pretrained Models on Task-based Compositional Generalization

This guide explains how to fine-tune pretrained language models (e.g., Llama 8B) on the generated task-based compositional generalization benchmarks.

## Overview

We provide two main components:

1. **`src/models/pretrained.py`**: Module for loading pretrained models from HuggingFace
2. **`src/training/finetuning.py`**: Training loop for fine-tuning pretrained models
3. **`scripts/finetune_model.py`**: Standalone script to run fine-tuning

## Installation

Make sure you have the required dependencies installed:

```bash
pip install transformers torch torchvision accelerate datasets
pip install numpy pandas tqdm omegaconf
```

For Llama models, you may need to install additional requirements:

```bash
pip install bitsandbytes  # For 4-bit/8-bit quantization
```

Note: Accessing Llama models requires HuggingFace authentication. You need to:
1. Accept the model license on the [Llama model page](https://huggingface.co/meta-llama/Llama-2-7b-hf)
2. Authenticate with HuggingFace CLI:

```bash
huggingface-cli login
```

## Usage

### Basic Usage

Fine-tune a GPT-2 model on the generated data:

```bash
python scripts/finetune_model.py \
    --config config/gen/conf.yaml \
    --model gpt2 \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 5e-5
```

### Fine-tuning Llama 7B

For larger models like Llama 7B, use smaller batch sizes and gradient accumulation:

```bash
python scripts/finetune_model.py \
    --config config/gen/conf.yaml \
    --model meta-llama/Llama-2-7b-hf \
    --epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --cache_dir ./cache/models
```

### Using Programmatic API

You can also use the modules directly in Python:

```python
from src.models.pretrained import PretrainedModelLoader
from src.training.finetuning import FineTuner

# Load a pretrained model
loader = PretrainedModelLoader(
    model_name="gpt2",
    device="cuda"
)
model, tokenizer, config = loader.load_model_and_tokenizer()

# Fine-tune the model
finetuner = FineTuner(
    model_name="gpt2",
    data_path="./data/diverse/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_7/direct/combination_6",
    output_dir="./checkpoints/gpt2",
    device="cuda",
    mode="direct"
)

finetuner.load_model_and_tokenizer()
finetuner.train(
    num_epochs=3,
    batch_size=4,
    learning_rate=5e-5
)
```

## Arguments

### Model Selection

- `--model`: HuggingFace model name
  - `gpt2`: GPT-2 small (117M parameters, good for testing)
  - `meta-llama/Llama-2-7b-hf`: Llama 2 7B (requires authentication)
  - `meta-llama/Llama-2-13b-hf`: Llama 2 13B (requires authentication)

### Data Configuration

- `--data_path`: Path to the generated data directory
  - Default: Automatically constructed from config
- `--mode`: Data format mode
  - `direct`: Direct I/O format
  - `step_by_step`: Step-by-step reasoning format
  - `curriculum`: Curriculum learning format

### Training Configuration

- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size (default: 4, use smaller for large models)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--weight_decay`: Weight decay for regularization (default: 0.01)
- `--gradient_accumulation_steps`: Accumulate gradients over N steps (default: 1)
- `--save_steps`: Save checkpoint every N steps (default: 500)
- `--eval_steps`: Evaluate every N steps (default: 500)
- `--max_grad_norm`: Maximum gradient norm for clipping (default: 1.0)

### Hardware Configuration

- `--device`: Device to use (`cuda` or `cpu`)
- `--cache_dir`: Directory to cache downloaded models (default: `./cache/models`)
  - **Important**: If you encounter permission errors, specify a writable directory:
    ```bash
    --cache_dir /path/to/writable/cache
    ```
- `--seed`: Random seed for reproducibility

## Output Structure

Fine-tuning creates the following directory structure:

```
checkpoints/
└── model_name/
    └── config/
        └── data_config/
            ├── checkpoints/
            │   ├── best/              # Best model checkpoint
            │   ├── step_500/          # Checkpoints at specific steps
            │   └── step_1000/
            ├── finetuning.log        # Training log
            ├── finetuning_run.log    # Run log
            └── config.yaml           # Configuration used
```

## Memory Considerations

Large models like Llama 7B require significant GPU memory. Here are some recommendations:

### For Llama 7B:
- **16GB GPU**: Use `--batch_size 1` with `--gradient_accumulation_steps 8`
- **24GB GPU**: Use `--batch_size 2` with `--gradient_accumulation_steps 4`
- **40GB+ GPU**: Use `--batch_size 4` with `--gradient_accumulation_steps 2`

### Memory Optimization Techniques:

1. **Gradient Checkpointing**: Already enabled by default
2. **Mixed Precision**: Use `--torch_dtype float16` (default)
3. **8-bit Training**: For very limited memory, modify the loader to use `BitsAndBytesConfig`

Example for 8-bit training:

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

loader = PretrainedModelLoader(
    model_name="meta-llama/Llama-2-7b-hf",
    model_kwargs={"quantization_config": quantization_config}
)
```

## Loading Fine-tuned Models

After fine-tuning, load your model:

```python
from src.models.pretrained import PretrainedModelLoader

# Load from checkpoint
loader = PretrainedModelLoader(
    model_name="./checkpoints/gpt2/best",
    device="cuda"
)
model, tokenizer, config = loader.load_model_and_tokenizer()

# Use the model
inputs = tokenizer.encode("Your input here", return_tensors="pt")
outputs = model.generate(inputs, max_length=128)
print(tokenizer.decode(outputs[0]))
```

## Troubleshooting

### Cache Permission Errors

If you encounter permission errors like "The process doesn't have correct read-write permission", this means HuggingFace is trying to use a system-wide cache directory.

**Solution**: Specify a writable cache directory:
```bash
python scripts/finetune_model.py \
    --model gpt2 \
    --cache_dir /project/pi_jensen_umass_edu/ppruthi_umass_edu/task_based_compositional_generalization/cache/models \
    ...
```

The cache directory defaults to `./cache/models` in the project root, which should be writable.

### Out of Memory Errors

1. Reduce `--batch_size`
2. Increase `--gradient_accumulation_steps`
3. Use a smaller model (e.g., `gpt2` instead of `meta-llama/Llama-2-7b-hf`)

### Slow Training

1. Use mixed precision training (already enabled by default)
2. Increase `--batch_size` if memory permits
3. Use multiple GPUs with `torch.nn.DataParallel` or `accelerate`

### Model Loading Errors

1. Ensure you're authenticated for Llama models:
   ```bash
   huggingface-cli login
   ```
2. Check that the model name is correct
3. Verify you have enough disk space for model weights
4. If using Llama, make sure you accepted the model license on HuggingFace

## Evaluation

After fine-tuning, you can evaluate the model using the existing evaluation scripts:

```bash
python scripts/evaluate_model.py \
    --checkpoint ./checkpoints/gpt2/best \
    --config config/gen/conf.yaml
```

## Citation

If you use this fine-tuning framework, please cite:

```bibtex
@misc{task_compositional_generalization,
    title={Task-Based Compositional Generalization},
    author={Your Name},
    year={2024}
}
```

