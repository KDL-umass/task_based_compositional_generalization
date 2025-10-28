#!/usr/bin/env python
"""
Fine-tuning script for pretrained models on task-based compositional generalization data.

Usage:
    python scripts/finetune_model.py --model llama3 --batch_size 4 --gradient_accumulation_steps 1

Arguments match train_model.py for consistency:
    --prompt_mode: direct, step_by_step, or curriculum
    --prompt_length: fixed or variable
    --train_split: combination_K, permutation_K, etc.
    --epochs, --batch_size, --learning_rate, etc.
"""

import argparse
import logging
import os

import torch
from omegaconf import OmegaConf
import yaml

from src.data_generation.init import read_config, set_seed
from src.training.finetuning import FineTuner

ROOT_DIR = "/project/pi_jensen_umass_edu/ppruthi_umass_edu/task_based_compositional_generalization"


def build_and_update_config(args):
    """
    Build and update configuration from arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Updated configuration object
    """
    cfg = read_config(f"{ROOT_DIR}/config/train/conf.yaml")
    
    # Calculate derived values
    n_alphabets_seq_len_fn_len_task_max_length = (
        "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
            args.n_alphabets, args.seq_len, args.n_functions, args.task_max_length
        )
    )
    
    data_path = "{}/data/{}/{}/{}/{}/{}".format(
        ROOT_DIR,
        args.function_type,
        args.prompt_length,
        n_alphabets_seq_len_fn_len_task_max_length,
        args.prompt_mode,
        args.train_split,
    )
    
    # Update config using OmegaConf.merge for cleaner updates
    updates = OmegaConf.create({
        "tag": args.prompt_mode,
        "prompt_length": args.prompt_length,
        "train_split": args.train_split,
        "epochs": args.epochs,
        "task_max_length": args.task_max_length,
        "function_type": args.function_type,
        "seed": args.seed,
        "n_alphabets": args.n_alphabets,
        "seq_len": args.seq_len,
        "net": {"prompt_length": args.prompt_length},
        "data": {
            "path": data_path,
            "n_alphabets_seq_len_fn_len_task_max_length": n_alphabets_seq_len_fn_len_task_max_length,
        },
        "function": {
            "type": args.function_type,
            "n_functions": args.n_functions,
            "split": {"strategy": args.train_split},
        },
    })
    
    return OmegaConf.merge(cfg, updates)


def setup_logging(output_dir):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, "finetuning_run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def _add_data_arguments(parser):
    """Add data configuration arguments matching train_model.py."""
    parser.add_argument("--prompt_mode", type=str, default="direct",
                       help="Prompt mode: direct, step_by_step, or curriculum")
    parser.add_argument("--prompt_length", type=str, default="fixed",
                       help="Prompt length: fixed or variable")
    parser.add_argument("--train_split", type=str, default="combination_6",
                       help="Training split strategy")
    parser.add_argument("--function_type", type=str, default="uniform",
                       help="Function type: uniform or diverse")
    parser.add_argument("--n_alphabets", type=int, default=26)
    parser.add_argument("--seq_len", type=int, default=6)
    parser.add_argument("--n_functions", type=int, default=6)
    parser.add_argument("--task_max_length", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)


def _add_training_arguments(parser):
    """Add training hyperparameter arguments."""
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)


def _add_model_arguments(parser):
    """Add model-specific arguments."""
    parser.add_argument("--model", type=str, default="llama3", required=True,
                       help="HuggingFace model name")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Model cache directory")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                       choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--bf16", action="store_true")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained models")
    
    _add_data_arguments(parser)
    _add_training_arguments(parser)
    _add_model_arguments(parser)
    
    args = parser.parse_args()
    
    # Build and apply config
    cfg = build_and_update_config(args)
    
    set_seed(cfg.seed)

    # Set default cache directory if not provided
    if args.cache_dir is None:
        args.cache_dir = os.path.join(ROOT_DIR, "cache", "models")
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)

    # Set data path and output directory
    args.data_path = cfg.data.path
    model_short = args.model.split("/")[-1].replace("-", "_")
    args.output_dir = os.path.join(
        ROOT_DIR,
        "checkpoints",
        model_short,
        cfg.function_type,
        cfg.data.n_alphabets_seq_len_fn_len_task_max_length,
        cfg.tag,
        cfg.train_split,
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(args.output_dir)

    logger.info("=" * 80)
    logger.info("Fine-tuning Configuration")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Prompt mode: {args.prompt_mode}")
    logger.info(f"Prompt length: {cfg.prompt_length}")
    logger.info(f"Train split: {args.train_split}")
    logger.info(f"Function type: {args.function_type}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"Warmup steps: {args.warmup_steps}")
    logger.info(f"Torch dtype: {args.torch_dtype}")
    logger.info(f"Compile model: {args.compile}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info("=" * 80)

    # Save config
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        yaml.dump(
            {
                "model": args.model,
                "data_path": cfg.data.path,
                "prompt_mode": args.prompt_mode,
                "prompt_length": args.prompt_length,
                "train_split": args.train_split,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "device": args.device,
                "function_type": args.function_type,
                "n_alphabets": args.n_alphabets,
                "seq_len": args.seq_len,
                "n_functions": args.n_functions,
                "task_max_length": args.task_max_length,
                "seed": args.seed,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "warmup_steps": args.warmup_steps,
                "save_steps": args.save_steps,
                "eval_steps": args.eval_steps,
                "max_grad_norm": args.max_grad_norm,
                "torch_dtype": args.torch_dtype,
                "compile": args.compile,
                "bf16": args.bf16,
            },
            f,
        )

    # Create fine-tuner
    finetuner = FineTuner(
        model_name=args.model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
        mode=args.prompt_mode,
        cfg=cfg,  # Pass config for data loading
    )

    # Load model
    finetuner.load_model_and_tokenizer()

    # Train
    finetuner.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
    )

    logger.info("Fine-tuning completed successfully!")


if __name__ == "__main__":
    main()

