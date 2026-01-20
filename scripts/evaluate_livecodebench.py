"""
LiveCodeBench evaluation script.
"""
import argparse
import logging

from init import ROOT_DIR, read_config
from src.evaluation.livecodebench_evaluator import LiveCodeBenchEvaluator


def main(cfg):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("livecodebench_eval")
    evaluator = LiveCodeBenchEvaluator(cfg, logger=logger)
    results = evaluator.evaluate()
    output_path = evaluator.save_results(results)
    logger.info("Saved results to: %s", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=f"{ROOT_DIR}/config/eval/livecodebench.yaml",
    )
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--holdout_size", type=int, default=None)
    parser.add_argument("--data_seed", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--include_fewshot", type=bool, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg = read_config(args.config)
    if args.model_name is not None:
        cfg.model_name = args.model_name
    if args.split is not None:
        cfg.split = args.split
    if args.holdout_size is not None:
        cfg.holdout_size = args.holdout_size
    if args.data_seed is not None:
        cfg.data_seed = args.data_seed
    if args.data_path is not None:
        cfg.data_path = args.data_path
    if args.max_samples is not None:
        cfg.max_samples = args.max_samples
    if args.max_new_tokens is not None:
        cfg.max_new_tokens = args.max_new_tokens
    if args.temperature is not None:
        cfg.temperature = args.temperature
    if args.top_p is not None:
        cfg.top_p = args.top_p
    if args.include_fewshot is not None:
        cfg.include_fewshot = args.include_fewshot
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size

    main(cfg)
