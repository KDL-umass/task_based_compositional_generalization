"""
Fine-tuning script for pretrained language models on task-based compositional generalization.

This script fine-tunes pretrained models (e.g., Llama) on the generated synthetic datasets
for task-based compositional generalization.
"""

import argparse
import json
import logging
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    get_linear_schedule_with_warmup,
)
import pickle

from src.data_generation.generator import (
    get_trainLoader_with_mapping,
    get_evalLoaders_with_mapping,
   
)
from src.models.pretrained import load_llama3_8b, load_gpt_oss_20b

ROOT_DIR = "/project/pi_jensen_umass_edu/ppruthi_umass_edu/task_based_compositional_generalization"


class FineTuner:
    """Handle fine-tuning of pretrained models."""

    def __init__(
        self,
        model_name: str,
        data_path: str,
        output_dir: str,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        mode: str = "direct",
        torch_dtype: str = "bfloat16",
        cfg=None,
    ):
        """
        Initialize the fine-tuner.

        Args:
            model_name: HuggingFace model name
            data_path: Path to data directory
            output_dir: Directory to save checkpoints
            cache_dir: Directory to cache downloaded models
            device: Device to use
            mode: Data mode ('direct', 'step_by_step', or 'curriculum')
            cfg: Configuration object (needed for data loading)
        """
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.device = device
        self.mode = mode
        self.cfg = cfg
        if torch_dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            self.torch_dtype = torch.float16
        elif torch_dtype == "float32":
            self.torch_dtype = torch.float32
        else:
            raise ValueError(f"Unknown torch dtype: {torch_dtype}")
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Setup logging
        self.setup_logging()

        self.model = None
        self.tokenizer = None
        self.config = None
        self.token_map = None  # Mapping from synthetic vocab to model vocab

    def setup_logging(self):
        """Setup logging configuration."""
        log_file = os.path.join(self.output_dir, "finetuning.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def load_model_and_tokenizer(self):
        """Load the pretrained model and tokenizer."""
        self.logger.info(f"Loading model: {self.model_name}")
        if self.model_name == "llama3":
            self.model, self.tokenizer, self.config = load_llama3_8b(torch_dtype=self.torch_dtype)
        elif self.model_name == "gpt":
            self.model, self.tokenizer, self.config = load_gpt_oss_20b(torch_dtype=self.torch_dtype)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

        self.logger.info(f"Model loaded successfully!")
        # Prepare token mapping
        self._setup_token_mapping()

    def _setup_token_mapping(self):
        """Setup mapping from synthetic vocab to model vocab."""
        # Load token mappings from data directory
        token_idx_path = os.path.join(self.data_path, "token_idx.pkl")
        with open(token_idx_path, "rb") as f:
            token_idx = pickle.load(f)
           
        # Create mapping from synthetic indices to model tokens
        self.token_map = {}
        special_tokens = []
        for token, idx in token_idx.items():
            # Get the token ID from the model's tokenizer
            model_token_id = self.tokenizer.convert_tokens_to_ids(token)
            if model_token_id is None:
                special_tokens.append(token)
            self.token_map[idx] = model_token_id

        num_added = self.tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens
        })
        print(f"Added {num_added} special tokens")
        print(f"Special tokens: {special_tokens}")
        self.model.resize_token_embeddings(len(self.tokenizer))

        for token, idx in token_idx.items():
            if self.token_map[idx] is None:
                self.token_map[idx] = self.tokenizer.convert_tokens_to_ids(token)
        print(f"Token map: {self.token_map}")

    def _convert_tokens_to_model_format(self, tokens):
        """
        Convert tokens to model format.
        """
        return torch.tensor([self.token_map[token] for token in tokens])
       

    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        save_steps: int = 500,
        eval_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        """
        Train the model.

        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            warmup_steps: Number of warmup steps for learning rate
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps
            gradient_accumulation_steps: Accumulate gradients over N steps
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.logger.info("Starting training...")
        self.logger.info(f"  Model: {self.model_name}")
        self.logger.info(f"  Epochs: {num_epochs}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Learning rate: {learning_rate}")

        # Create data loaders using existing infrastructure
        # Update cfg for data loading
        self.cfg.data.path = self.data_path
        self.cfg.data.batch_size = batch_size
        self.cfg.data.num_workers = 4
        self.cfg.tag = self.mode
        
        # Use loaders with token mapping applied
        train_loader = get_trainLoader_with_mapping(self.cfg, self.token_map)
        eval_loaders = get_evalLoaders_with_mapping(self.cfg, self.token_map)
        
        # eval_loaders is a list: [train_loader, test_loader, train_heldout_loader]
        # Use test_loader (index 1) for validation
        val_loader = eval_loaders[1]

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Setup scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        # Training loop
        self.model.train()
        global_step = 0
        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0.0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}",
                total=len(train_loader),
            )

            for step, (dat, targets) in enumerate(progress_bar):
                # Move to device (tokens already converted to model format)
                dat = dat.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=dat)
                logits = outputs.logits
                
                # Compute loss
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                # Scale loss by gradient accumulation
                loss = loss / gradient_accumulation_steps

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm
                )

                # Optimizer step
                if (step + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

                # Evaluation
                if global_step % eval_steps == 0:
                    val_loss = self.evaluate(val_loader)
                    self.logger.info(f"Step {global_step}: Validation loss = {val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(global_step, is_best=True)

                # Save checkpoint
                if global_step % save_steps == 0:
                    self.save_checkpoint(global_step)

            avg_loss = epoch_loss / len(train_loader)
            self.logger.info(f"Average training loss: {avg_loss:.4f}")

        self.logger.info("Training completed!")

    def evaluate(self, val_loader):
        """Evaluate the model on validation set."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for dat, targets in val_loader:
                # Move to device (tokens already converted to model format)
                dat = dat.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=dat)
                logits = outputs.logits
                
                # Compute loss
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        self.model.train()
        return avg_loss

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints", f"step_{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        if is_best:
            best_dir = os.path.join(self.output_dir, "checkpoints", "best")
            os.makedirs(best_dir, exist_ok=True)
            self.model.save_pretrained(best_dir)
            self.tokenizer.save_pretrained(best_dir)
            self.logger.info(f"Saved best model to {best_dir}")

        self.logger.info(f"Saved checkpoint to {checkpoint_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained model on synthetic data"
    )
    parser.add_argument(
        "--model_name", type=str, default="llama3", help="HuggingFace model name"
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to data")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument("--mode", type=str, default="direct", help="Data mode")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Model cache directory"
    )
    parser.add_argument("--torch_dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"], help="Data type for the model")  
    args = parser.parse_args()

    # Create fine-tuner
    torch_dtype = getattr(torch, args.torch_dtype)
    print(f"Using torch dtype: {torch_dtype}")
    finetuner = FineTuner(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
        mode=args.mode,
        torch_dtype=torch_dtype,
    )

    # Load model
    finetuner.load_model_and_tokenizer()

    # Train
    finetuner.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()

