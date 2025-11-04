"""
Evaluation script
"""

import argparse
import os

import numpy as np
import torch

from init import read_config, set_seed, ROOT_DIR, MODELS_ROOT_DIR
# from src.evaluation.evaluator import Evaluator
from src.models.nanogpt import nanoGPT
from src.models.pretrained import load_llama3_8b, load_gpt_oss_20b, load_granite_2b
from src.utils.logging_utils import setup_evaluation_logging

# see if cuda is available
if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    DEVICE = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    DEVICE = torch.device("cpu")


def load_net(fname):
    ckpt = torch.load(fname, weights_only=False, map_location=DEVICE)
    net_cfg = ckpt["config"]

    net = nanoGPT(net_cfg.net)

    net.load_state_dict(ckpt["net"])
    return net, net_cfg

def load_pretrained_model_from_ckpt(model_name, ckpt_path):
    if model_name == "llama3":
        model, tokenizer, config = load_llama3_8b()
    elif model_name == "gpt2":
        model, tokenizer, config = load_gpt_oss_20b()
    elif model_name == "granite":
        model, tokenizer, config = load_granite_2b()
    else:
        raise ValueError(f"Model {model_name} not supported")
    model.load_state_dict(torch.load(ckpt_path, weights_only=False, map_location=DEVICE))
    return model, tokenizer, config


def fetch_dirs(cfg, i):
    ckpt_dir = "{}/models/ckpts/{}/{}/{}/{}/{}".format(
        MODELS_ROOT_DIR,
        cfg.function_type,
        cfg.prompt_length,
        cfg.data_n_alphabets_seq_len_fn_len_max_task_length,
        cfg.prompt_mode,
        cfg.train_split,
    )
    if not cfg.pretrained:
        ckpt_dir += "/{}/{}/seed_{}".format(
            cfg.pos_embedding_type,
            cfg.nheads_nlayers,
            cfg.seed,
        )
    else:
        ckpt_dir += "/{}".format(cfg.model_name)
    if not os.path.exists(ckpt_dir):
        raise ValueError("Checkpoint directory does not exist: {}".format(ckpt_dir))

    def itr(ck):
        return int((ck.split("_")[-1]).split(".")[0])

    all_dirs = os.listdir(ckpt_dir)
    all_dirs = [os.path.join(ckpt_dir, d) for d in all_dirs if d.endswith(".pt")]
    all_dirs = [(itr(d), d) for d in all_dirs]
    all_dirs = sorted(all_dirs)

    reduced_alldirs = []
    for it, cdir in all_dirs:
        reduced_alldirs.append((it, cdir))

    return reduced_alldirs


def main(cfg):
    print("Running evaluation with the following configuration:")
    print(cfg)
    set_seed(cfg.seed)
    for i in range(cfg.num_runs):
        sorted_dirs = fetch_dirs(cfg, i)
        print("Sorted dirs: ", sorted_dirs)
        if cfg.pretrained:
            model, tokenizer, config = load_pretrained_model_from_ckpt(cfg.model_name, sorted_dirs[0][1])
            net = model
            net_cfg = config
        else:
            _, net_cfg = load_net(sorted_dirs[0][1])
    
        # set the logger from utils
        logger = setup_evaluation_logging(cfg)
        # evaluator = Evaluator(cfg, logger)
        # metrics = evaluator.evaluate(net)
        # for k, v in metrics.items():
        #     result_dict = v
        #     evaluator.save_accs(cfg, metrics, logger)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_mode", type=str, default="direct", help="step or direct"
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="combination_6",
        help="Training split",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="combination_6",
        help="Model evaluation split",
    )
    parser.add_argument(
        "--nheads_nlayers",
        type=str,
        default="nh12_nl12",
        help="number of heads and layers",
    )
    parser.add_argument(
        "--pos_embedding_type", type=str, default="rel_global", help="abs or rel_global"
    )
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs")
    parser.add_argument(
        "--function_type", type=str, default="uniform", help="uniform or diverse"
    )
    parser.add_argument(
        "--task_max_length", type=int, default=6, help="max task length"
    )
    
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # over-ride some of the arguments based on the run-time args
    cfg = read_config(f"{ROOT_DIR}/config/eval/conf.yaml") 
    cfg.prompt_mode = args.prompt_mode
    cfg.train_split = args.train_split
    cfg.eval_split = args.eval_split
    cfg.nheads_nlayers = args.nheads_nlayers
    cfg.pos_embedding_type = args.pos_embedding_type
    cfg.num_runs = args.num_runs
    cfg.function_type = args.function_type
    cfg.seed = args.seed
    cfg.task_max_length = args.task_max_length
    cfg.data_n_alphabets_seq_len_fn_len_max_task_length = (
        "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
            cfg.n_alphabets, cfg.seq_len, cfg.n_functions, cfg.task_max_length
        )
    )
    cfg.data_path = f"{MODELS_ROOT_DIR}/data/{cfg.function_type}/{cfg.prompt_length}/{cfg.data_n_alphabets_seq_len_fn_len_max_task_length}/{cfg.prompt_mode}/{cfg.eval_split}"
    cfg.train_data_path = f"{MODELS_ROOT_DIR}/data/{cfg.function_type}/{cfg.prompt_length}/{cfg.data_n_alphabets_seq_len_fn_len_max_task_length}/{cfg.prompt_mode}/{cfg.train_split}"
    
    main(cfg)
