import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
import os

# Get sequence information
def get_seq_info(token_idx, sep_token, end_token, sample, function_type):
    """Extract sequence information"""
    seq_info = {}
    sep_idx = token_idx[sep_token]
    total_len = len(sample)

    sep_pos = np.where(sample == sep_idx)[0]

    last_sep_pos = sep_pos[-1]
    if function_type == "uniform":
        third_sep_pos = sep_pos[1]
    else:
        third_sep_pos = sep_pos[2]
    end_token_pos = np.where(sample == token_idx[end_token])[0]

    seq_info["last_sep_pos"] = last_sep_pos
    seq_info["prompt_pos_end"] = third_sep_pos + 1
    seq_info["end_pos"] = end_token_pos[0]
    extra_space_tokens = total_len - seq_info["end_pos"] - 1
    seq_info["new_len"] = total_len - seq_info["prompt_pos_end"] - extra_space_tokens

    return seq_info


def get_sep_pos(fpath, loader):
    token_idx = np.load(os.path.join(fpath, "token_idx.pkl"), allow_pickle=True)
    sep_idx = token_idx["<SEP>"]
    sep_pos = np.where(loader.dataset.data[0] == sep_idx)[0][-1]
    return sep_pos


def get_function_list(doc, sep_idx):
    sep_pos = np.where(doc == sep_idx)[0][0]
    doc_function = doc[1:sep_pos]
    return doc_function


def get_input_string(doc, sep_idx, function_type):
    if function_type == "diverse":
        third_sep_pos = np.where(doc == sep_idx)[0][2]
    else:
        third_sep_pos = np.where(doc == sep_idx)[0][1]
    first_sep_pos = np.where(doc == sep_idx)[0][0]
    input_string = doc[first_sep_pos + 1 : third_sep_pos]
    return input_string


def get_output_string(doc, sep_idx, end_token, function_type):
    if function_type == "diverse":
        third_sep_pos = np.where(doc == sep_idx)[0][2]
    else:
        third_sep_pos = np.where(doc == sep_idx)[0][1]
    end_token_pos = np.where(doc == end_token)[0][0]
    output_string = doc[third_sep_pos + 1 : end_token_pos]
    return output_string


