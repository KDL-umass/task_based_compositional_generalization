import numpy as np
from init import read_config, ROOT_DIR
import pickle
import argparse
from src.data.composition_generator.compositions import CompositionsGenerator
from src.data.loaders import get_data_loader, SyntheticDataset

class CompositionSequenceDistanceCalculator:
    def __init__(self, cfg):
        self.cfg = cfg

   

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="combination_6",
        help="Random split of permutations of size K: combination_K; \
        Systematic split of permutations of size K with one relative order in training: permutation_K; \
        Systematic split of size K with T relative orders in training: permutation_T_K",
    )
    parser.add_argument(
        "--task_max_length",
        type=int,
        default=6,
        help="compositional task max length"
    )
    parser.add_argument(
        "--function_type", type=str, default="diverse", help="uniform or diverse"
    )
    parser.add_argument(
        "--mode", type=str, default="direct", help="direct or step_by_step"
    )
    args = parser.parse_args()
    cfg = read_config("./config/gen/conf.yaml")
    ce_metric_calculation = CEMetricCalculator(cfg)
    cfg.split_strategy = args.split_strategy
    cfg.task_max_length = args.task_max_length
    cfg.function_type = args.function_type
    compositions_generator = CompositionsGenerator(cfg)
    train_functions, test_functions, functions_info = compositions_generator.get_train_test_compositions()
        
    all_functions = train_functions + test_functions
    print(f"All functions: {len(all_functions)}")
    
    ce_metric, ce_metric_dict = ce_metric_calculation.calculate_ce_metric(all_functions, compositions_generator, N_samples=1000, final_output=args.final_output)
    print(f"CE metric: {ce_metric}")
    
    # save ce_metric_dict in data directory
    with open(f"ce_metric_dict_{cfg.split_strategy}_final_output_{args.final_output}.pkl", "wb") as f:
        pickle.dump(ce_metric_dict, f)
    
if __name__ == "__main__":
    main()