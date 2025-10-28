"""
Module for loading pretrained language models from HuggingFace.

This module provides utilities to download and load pretrained transformer models
like Llama, GPT-2, etc. for fine-tuning on task-based compositional generalization.
"""

import os
from typing import Optional
from init import ROOT_DIR

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

LLAMA3_MODEL_NAME = "meta-llama/Llama-3.1-8b"
GPT_OSS_MODEL_NAME = "openai/gpt-oss-20b"
# Set environment variables to control cache locations
def set_cache_env_vars(model_name: str):
    """
    Set environment variables to force HuggingFace to use the specified cache directory.
    
    Args:
        cache_dir: Directory to cache models
    """
    # Set HF_HOME to control all HuggingFace cache locations
    if model_name == LLAMA3_MODEL_NAME:
        postfix = "llama3"
    elif model_name == GPT_OSS_MODEL_NAME:
        postfix = "gpt"
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    os.environ['HF_HOME'] = f"/datasets/ai/{postfix}"
    os.environ['HF_DATASETS_CACHE'] = os.path.join(ROOT_DIR, postfix)
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(ROOT_DIR, postfix, 'transformers')
    os.environ['HF_HUB_CACHE'] = os.path.join(ROOT_DIR, postfix, 'hub')

    print(f"Set cache environment variables to: {ROOT_DIR}")


class PretrainedModelLoader:
    """Load pretrained language models from HuggingFace."""

    def __init__(
        self,
        model_name: str = LLAMA3_MODEL_NAME,
        cache_dir: Optional[str] = None,
        device: str = "cuda",
        **model_kwargs,
    ):
        """
        Initialize the pretrained model loader.

        Args:
            model_name: HuggingFace model name or path
            cache_dir: Directory to cache downloaded models
            device: Device to load model on ('cuda' or 'cpu')
            **model_kwargs: Additional arguments to pass to model loading
        """
        self.model_name = model_name
            
        self.cache_dir = cache_dir or os.path.join(ROOT_DIR, "cache", "models")
        self.device = device
        self.model_kwargs = model_kwargs

        set_cache_env_vars(self.model_name)

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.config = None

    def load_model_and_tokenizer(self, torch_dtype=torch.bfloat16):
        """
        Load the model and tokenizer from HuggingFace.

        Args:
            torch_dtype: Data type for the model (default: bfloat16 for efficiency)

        Returns:
            Tuple of (model, tokenizer, config)
        """
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")
        print(f"Using dtype: {torch_dtype}")

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load config
        print("Loading config...")
        self.config = AutoConfig.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )

        # Load model
        print("Loading model...")
        model_kwargs = {
            **self.model_kwargs,
            "torch_dtype": torch_dtype,
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
        }

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        # Move model to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(self.device)
            print(f"Model moved to {self.device}")
        elif self.device == "cuda":
            print("CUDA not available, using CPU")
            self.device = "cpu"
            self.model = self.model.to("cpu")

        # Enable gradient checkpointing to save memory (optional)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")

        print(f"Model loaded successfully!")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        print(f"Model parameters: {self.count_parameters()}")

        return self.model, self.tokenizer, self.config

    def count_parameters(self, trainable_only: bool = False) -> int:
        """
        Count the number of parameters in the model.

        Args:
            trainable_only: If True, only count trainable parameters

        Returns:
            Number of parameters
        """
        if self.model is None:
            return 0

        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.model.parameters())

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {}

        return {
            "model_name": self.model_name,
            "device": self.device,
            "total_parameters": self.count_parameters(trainable_only=False),
            "trainable_parameters": self.count_parameters(trainable_only=True),
            "vocab_size": self.config.vocab_size if self.config else None,
            "max_position_embeddings": (
                self.config.max_position_embeddings if self.config else None
            ),
            "hidden_size": self.config.hidden_size if self.config else None,
            "num_layers": self.config.num_hidden_layers if self.config else None,
            "num_attention_heads": (
                self.config.num_attention_heads if self.config else None
            ),
        }

def load_pretrained_model(
    model_name: str = "meta-llama/Llama-3.1-8b",
    cache_dir: Optional[str] = None,
    device: str = "cuda",
    torch_dtype=torch.bfloat16,
    **kwargs,
):
    """
    Convenience function to load a pretrained model.

    Args:
        model_name: HuggingFace model name
        cache_dir: Directory to cache downloaded models
        device: Device to load model on
        torch_dtype: Data type for the model
        **kwargs: Additional arguments for model loading

    Returns:
        Tuple of (model, tokenizer, config)
    """
    loader = PretrainedModelLoader(
        model_name=model_name, cache_dir=cache_dir, device=device, **kwargs
    )
    model, tokenizer, config = loader.load_model_and_tokenizer(torch_dtype=torch_dtype)
    print(f"\nModel info:")
    for key, value in loader.get_model_info().items():
        print(f"  {key}: {value}")
    return model, tokenizer, config


def load_llama3_8b(device: str = "cuda", cache_dir: Optional[str] = None, torch_dtype=torch.bfloat16):
    """
    Load Llama 3 8B model from local path.

    Args:
        device: Device to load model on
        cache_dir: Directory to cache downloaded models

    Returns:
        Tuple of (model, tokenizer, config)
    """
    return load_pretrained_model(
        model_name=LLAMA3_MODEL_NAME,
        cache_dir=cache_dir,
        device=device,
        torch_dtype=torch_dtype,
    )


def load_gpt_oss_20b(device: str = "cuda", cache_dir: Optional[str] = None, torch_dtype=torch.bfloat16):
    """
    Load GPT OSS 20B model from local path.

    Args:
        device: Device to load model on
        cache_dir: Directory to cache downloaded models

    Returns:
        Tuple of (model, tokenizer, config)
    """
    return load_pretrained_model(
        model_name=GPT_OSS_MODEL_NAME,
        cache_dir=cache_dir,
        device=device,
        torch_dtype=torch_dtype,
    )


if __name__ == "__main__":
    # Example usage
    print("Testing model loader...")
    print("=" * 50)
    model, tokenizer, config = load_llama3_8b()
    # model, tokenizer, config = load_gpt_oss_20b()