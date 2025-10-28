import argparse
from init import read_config, set_seed, ROOT_DIR
from src.data.generator import SyntheticDataGenerator

def main(cfg):
    set_seed(cfg.seed)
    # Create synthetic data generator
    synthetic_data_generator = SyntheticDataGenerator(cfg)
    # Generate corpus
    synthetic_data_generator.generate_corpus()
    # Store data
    synthetic_data_generator.store_data()


if __name__ == "__main__":
    # Set config in the yaml files

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="combination_6",
        help="Random split of permutations of size K: combination_K; \
        Systematic split of permutations of size K with one relative order in training: permutation_K; \
        Systematic split of size K with T relative orders in training: permutation_T_K",
        required=True,
    )
    parser.add_argument(
        "--task_max_length",
        type=int,
        default=7,
        help="compositional task max length",
        required=True,
    )
    parser.add_argument(
        "--function_type", type=str, default="uniform", help="uniform or diverse"
    )
    
    args = parser.parse_args()
    # Read config file
    cfg_path = "{}/config/gen/conf.yaml".format(ROOT_DIR)
    # read the config file
    cfg = read_config(cfg_path)
    cfg.split_strategy = args.split_strategy
    cfg.task_max_length = args.task_max_length
    cfg.function_type = args.function_type
    main(cfg)
