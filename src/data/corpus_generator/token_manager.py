"""Token management for synthetic data generation."""
import numpy as np
import logging
from src.data.synthetic_sequence_generator.constants import SPECIAL_TOKENS
import os

class TokenManager:
    """Handles token initialization, encoding, and decoding."""
    
    def __init__(self, n_alphabets, function_dict):
        self.n_alphabets = n_alphabets
        self.special_tokens = SPECIAL_TOKENS
        self.function_dict = function_dict
        self.token = {}
        self.token_idx = {}
        self.logger = logging.getLogger(__name__)
        
        # Special token attributes
        self.start_token = self.special_tokens["START"]
        self.space_token = self.special_tokens["SPACE"]
        self.sep_token = self.special_tokens["SEP"]
        self.null_token = self.special_tokens["NULL"]
        self.end_token = self.special_tokens["END"]
        
    def init_tokens(self):
        """Initialize alphabet, special, and function tokens."""
        # Create alphabet tokens (a-z)
        for i in range(self.n_alphabets):
            self.token[i] = chr(i + 97)
            self.token_idx[chr(i + 97)] = i
        
        # Add special tokens
        sp_token_count = self._add_special_tokens()
        
        # Add function tokens
        self._add_function_tokens(sp_token_count)
        
        # Create index arrays for quick access
        self._create_index_arrays()
        
        self.logger.info("Tokens: {}".format(self.token))
        self.logger.info("Token indices: {}".format(self.token_idx))
        
    def _add_special_tokens(self):
        """Add special tokens."""
        sp_token_count = 0
        for token in self.special_tokens:
            self.token[self.n_alphabets + sp_token_count] = token
            self.token_idx[token] = self.n_alphabets + sp_token_count
            sp_token_count += 1
        return sp_token_count
    
    def _add_function_tokens(self, offset):
        """Add function tokens."""
        count = 0
        for token in self.function_dict.keys():
            self.token[self.n_alphabets + offset + count] = token
            self.token_idx[token] = self.n_alphabets + offset + count
            count += 1
    
    def _create_index_arrays(self):
        """Create numpy arrays for frequently used token indices."""
        self.start_idx = np.array([self.token_idx[self.start_token]])
        self.sep_idx = np.array([self.token_idx[self.sep_token]])
        self.end_idx = np.array([self.token_idx[self.end_token]])
        
        self.space_idx = np.array([self.token_idx[self.space_token]])
        self.null_idx = np.array([self.token_idx[self.null_token]])
    
    def decode(self, token_indices, return_list=False):
        """Decode token indices to human-readable string."""
        txt_list = []
        for i in token_indices:
            if i in self.token:
                txt_list.append(self.token[i])
                if not return_list:
                    txt_list.append(" ")
        
        # Remove last space
        if txt_list and txt_list[-1] == " ":
            txt_list = txt_list[:-1]
            
        return "".join(txt_list) if not return_list else txt_list
    
    def encode(self, txt):
        """Encode string to token indices."""
        return [self.token_idx[c] for c in txt]
    
    def get_task_token_indices(self, function_list):
        """Convert function list to token indices."""
        return np.array([self.token_idx[fn_name] for fn_name in function_list])
    
    def get_output_token_indices(self, outputs):
        """Convert output strings to token indices."""
        output_indices = []
        for output in outputs:
            output_idx = []
            if len(output) == 0:
                output_idx.append(self.token_idx["<NULL>"])
            else:
                output_idx = [self.token_idx[output[i]] for i in range(len(output))]
            output_indices.append(output_idx)
        return output_indices

    
# load token dictionary from file
class DictionaryLoader:
    def __init__(self, fpath):
        self.fpath = fpath
        self.token_fname = os.path.join(fpath, "token.pkl")
        self.token_idx_fname = os.path.join(fpath, "token_idx.pkl")
        self.token_dict = {}
        self.token_idx_dict = {}
        self.load_token_dict()

    def load_token_dict(self):
        token_dict = np.load(self.token_fname, allow_pickle=True)
        token_idx_dict = np.load(self.token_idx_fname, allow_pickle=True)
        self.token_dict = token_dict
        self.token_idx_dict = token_idx_dict

    def get_vocab_len(self):
        return len(self.token_idx_dict)