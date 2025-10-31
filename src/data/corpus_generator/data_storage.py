"""Data storage and directory management."""
import os
import pickle
import json
import logging
from omegaconf import OmegaConf
import numpy as np
from src.utils.logging_utils import setup_data_logging

class DataStorage:
    """Handles file I/O operations and directory structure."""
    
    def __init__(self, cfg, root_dir):
        self.cfg = cfg
        self.root_dir = root_dir
        self.dir_flag = cfg.split_strategy
        self._setup_directory_paths()
        
    def _setup_directory_paths(self):
        """Compute and create directory paths."""
        base_name = "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
            self.cfg.n_alphabets,
            self.cfg.seq_len,
            self.cfg.n_functions,
            self.cfg.task_max_length,
        )
        
        base_path = "{}/data/{}/{}/{}".format(
            self.root_dir,
            self.cfg.function_type,
            self.cfg.prompt_length,
            base_name,
        )
        
        self.step_fdir = "{}/step_by_step/{}".format(base_path, self.dir_flag)
        self.direct_fdir = "{}/direct/{}".format(base_path, self.dir_flag)
        
        # Create directories
        os.makedirs(self.step_fdir, exist_ok=True)
        os.makedirs(self.direct_fdir, exist_ok=True)    
        
    def setup_logging(self):
        """Initialize logging configuration using centralized logging utility."""
        print("data_dir", self.direct_fdir)
        
        logger = setup_data_logging(
            root_dir=self.root_dir,
            function_type=self.cfg.function_type,
            n_alphabets=self.cfg.n_alphabets,
            seq_len=self.cfg.seq_len,
            n_functions=self.cfg.n_functions,
            task_max_length=self.cfg.task_max_length,
            prompt_length=self.cfg.prompt_length,
            dir_flag=self.dir_flag,
            log_filename="data.log",
        )
        
        return logger
    
    def store_data(self, corpus, token, token_idx, functions_info):
        """Store all generated data to disk."""
        modes = ["step_by_step", "direct"]
        
        for mode in modes:
            mode_dir = self._get_mode_directory(mode)
            self._save_mode_data(mode, mode_dir, corpus, token, 
                               token_idx, functions_info)
    
    def _get_mode_directory(self, mode):
        """Get directory path for a specific mode."""
        if mode == "step_by_step":
            return self.step_fdir
        elif mode == "direct":
            return self.direct_fdir
    
    def _save_mode_data(self, mode, mode_dir, corpus, token, 
                       token_idx, functions_info):
        """Save data for a specific mode."""
        
        
        os.makedirs(mode_dir, exist_ok=True)
        
        # Save token mappings
        pickle.dump(token_idx, open(mode_dir + "/token_idx.pkl", "wb"))
        pickle.dump(token, open(mode_dir + "/token.pkl", "wb"))
        pickle.dump(functions_info, open(mode_dir + "/functions_info.pkl", "wb"))
        
        # Save corpus data
        np.save(mode_dir + "/train_{}_corpus.npy".format(mode),
                corpus["train_" + mode])
        np.save(mode_dir + "/test_{}_corpus.npy".format(mode),
                corpus["test_" + mode])
        np.save(mode_dir + "/train_heldout_{}_corpus.npy".format(mode),
                corpus["train_heldout_" + mode])
        
        # Save config
        config_dict = self._get_config_for_mode(mode)
        json.dump(config_dict, open(mode_dir + "/config.json", "w"), indent=4)
    
    def _get_config_for_mode(self, mode):
        """Get configuration dictionary for a specific mode."""
        cfg_copy = self.cfg.copy()
        cfg_copy["tag"] = mode
        
        if mode == "step_by_step":
            cfg_copy["direct"] = False
        elif mode == "direct":
            cfg_copy["direct"] = True
            
        return OmegaConf.to_container(cfg_copy, resolve=True)