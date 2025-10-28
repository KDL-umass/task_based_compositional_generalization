from src.data_generation.init import read_config
import argparse
from src.data_generation.ce_metric_calculation import CEMetricCalculator
from src.data_generation.functions import CreateFunctions
import pickle
ROOT_DIR = "/project/pi_jensen_umass_edu/ppruthi_umass_edu/task_based_compositional_generalization"

def get_train_test_functions(cfg):
    function_obj = CreateFunctions(cfg)
    # Create functions
    train_functions, test_functions, functions_info = function_obj.get_train_functions()
    print(f"Train functions: {len(train_functions)}")
    print(f"Test functions: {len(test_functions)}")
    function_dict = function_obj.function_dict
    return train_functions, test_functions, function_dict

def main(cfg, N_samples):    
    train_functions, test_functions, function_dict = get_train_test_functions(cfg)
    ce_metric_calculation = CEMetricCalculator(cfg)
    ce_metric, ce_metric_dict = ce_metric_calculation.calculate_ce_metric(train_functions, test_functions, function_dict, N_samples=N_samples)
    print(f"CE metric: {ce_metric}")
    pickle.dump(ce_metric_dict, open(cfg.data_path + "/ce_metric.pkl", "wb"))
    
    # save ce_metric_dict in data directory 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_mode", type=str, default="direct", help="step or direct"
    )
    parser.add_argument(
        "--model_split",
        type=str,
        default="combination_2",
        help="Model training split",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="combination_2",
        help="Model evaluation split",
    )
    parser.add_argument(
        "--function_type", type=str, default="diverse", help="uniform or diverse"
    )
    parser.add_argument(
        "--task_max_length", type=int, default=2, help="max task length"
    )
    parser.add_argument(
        "--N_samples", type=int, default=1000, help="number of samples"
    )
    args = parser.parse_args()
    cfg = read_config("./config/eval/conf.yaml")

    # over-ride some of the arguments based on the run-time args
    cfg.model_split = args.model_split
    cfg.eval_split = args.eval_split
    cfg.task_max_length = args.task_max_length
    cfg.function.split.strategy = cfg.model_split 
    cfg.data_n_alphabets_seq_len_fn_len_max_task_length = (
        "nalph_{}_seqlen_{}_fnlen_{}_taskmaxlen_{}".format(
            cfg.n_alphabets, cfg.seq_len, cfg.function.n_functions, args.task_max_length
        )
    )
    
    cfg.data_path = f"{ROOT_DIR}/data/{args.function_type}/{cfg.prompt_length}/{cfg.data_n_alphabets_seq_len_fn_len_max_task_length}/{args.prompt_mode}/{args.eval_split}"

    main(cfg, args.N_samples)
