"""Centralized logging utilities for data generation, training, and evaluation."""
import logging
import os
from typing import Optional
from init import ROOT_DIR


def setup_logger(
    log_dir: str,
    log_filename: str = "data.log",
    logger_name: Optional[str] = None,
    level: int = logging.INFO,
    format_string: str = "%(asctime)s - %(levelname)s - %(message)s",
    filemode: str = "w",
    propagate: bool = False,
    add_stream_handler: bool = False,
) -> logging.Logger:
    """
    Set up a logger with file handler.
    
    Args:
        log_dir: Directory where the log file will be created
        log_filename: Name of the log file (default: "data.log")
        logger_name: Name for the logger (default: None, uses module name)
        level: Logging level (default: logging.INFO)
        format_string: Format string for log messages
        filemode: File mode ('w' for overwrite, 'a' for append)
        propagate: Whether to propagate logs to parent logger
        add_stream_handler: Whether to also log to stdout/stderr
        
    Returns:
        Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create file handler
    log_file_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_file_path, mode=filemode)
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Optionally add stream handler
    if add_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    # Prevent propagation to root logger (optional, but recommended)
    logger.propagate = propagate
    
    return logger

def get_directory_path(cfg: dict, key: str) -> str:
    log_path = os.path.join(ROOT_DIR, "logs")
    function_type = cfg.function_type
    base_name = "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
        cfg.n_alphabets,
        cfg.seq_len,
        cfg.n_functions,
        cfg.task_max_length,
    )
    prompt_mode = cfg.prompt_mode
    prompt_length = cfg.prompt_length
    # if train_split is present in cfg, use it, otherwise use split_strategy
    if hasattr(cfg, 'train_split'):
        train_split = cfg.train_split
    else:
        train_split = cfg.split_strategy

    if key == 'eval':
        pos_embedding_type = cfg.net.pos_embedding_type
        eval_split = cfg.eval_split
        log_file_dir = os.path.join(
            log_path,
            function_type,
            base_name,
            prompt_mode,
            prompt_length,
            f"model_{train_split}",
            f"eval_{eval_split}",
            pos_embedding_type,
            f"seed_{cfg.seed}",
        )
    elif key == 'train':
        pos_embedding_type = cfg.net.pos_embedding_type
        log_file_dir = os.path.join(
            log_path,
            function_type,
            base_name,
            prompt_mode,
            prompt_length,
            f"model_{train_split}",
            pos_embedding_type,
            f"seed_{cfg.seed}",
        )
    elif key == 'data':
        log_file_dir = os.path.join(
            log_path,
            function_type,
            base_name,
            prompt_mode,
            prompt_length,
            f"model_{train_split}",
        )
    return log_file_dir


def setup_data_logging(
    cfg: dict,
    log_filename: str = "data.log",
) -> logging.Logger:
    """
    Set up logging for data generation tasks.
    
    Creates a logger with the standard data generation directory structure:
    {root_dir}/logs/{function_type}/{base_name}/{tag}/{prompt_length}/model_{train_split}/{pos_embedding_type}/
    
    Args:
        cfg: Configuration dictionary
        log_filename: Name of the log file
        
    Returns:
        Configured logger instance
    """
    log_path = get_directory_path(cfg, key='data')
    return setup_logger(log_path, log_filename=log_filename)


def setup_training_logging(
        cfg: dict,
        log_filename: str = "train.log",
) -> logging.Logger:
    """
    Set up logging for training tasks.
    
    Creates a logger with the standard training directory structure:
    {ROOT_DIR}/logs/{function_type}/{base_name}/{tag}/{prompt_length}/model_{train_split}/{pos_embedding_type}/
    
    Args:
        cfg: Configuration dictionary
        log_filename: Name of the log file
        
    Returns:
        Configured logger instance
    """
    log_path = get_directory_path(cfg, key='train')
    return setup_logger(log_path, log_filename=log_filename)


def setup_evaluation_logging(
    cfg: dict,
    log_filename: str = "eval.log",
) -> logging.Logger:
    """
    Set up logging for evaluation tasks.
    
    Creates a logger with the standard evaluation directory structure:
    {log_path}/{base_name}/{tag}/{prompt_length}/model_{model_split}/eval_{eval_split}/{pos_embedding_type}/seed_{seed}/
    
    Args:
        cfg: Configuration dictionary
        log_filename: Name of the log file
        
    Returns:
        Configured logger instance
    """
    log_path = get_directory_path(cfg, key='eval')
    
    logger = setup_logger(log_path, log_filename=log_filename)
    logger.info("Initializing SyntheticEval...")
    logger.info(os.path.join(log_path, log_filename))
    
    return logger

