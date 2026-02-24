import numpy as np
from itertools import permutations
import random
from typing import List, Tuple, Dict
import argparse
from init import read_config
from src.data.composition_generator.compositions import CompositionsGenerator
import matplotlib.pyplot as plt
import pickle
import json

RESULTS_PATH = "/scratch4/workspace/ppruthi_umass_edu-CG/task_based_compositional_generalization/models/eval/"

class ModuleCoverageMetricCalculator:
    """
    Analyzer for measuring divergence between train/test splits of permutation data.
    Focuses on position coverage and pairwise adjacency patterns.
    """
    
    def __init__(self, K: int = 6, unique_tokens: List[int] = None, epsilon: float = 1e-10):
        self.K = K
        self.token_idx_map = {}
        for i, token in enumerate(unique_tokens):
            self.token_idx_map[token] = i
        self.epsilon = epsilon
    def calculate_position_distribution(self, functions: List) -> np.ndarray:
        """
        Calculate P(token at position) for all tokens and positions.
        Returns: K x K matrix where entry [i,j] = P(token i+1 at position j)
        """
        dist = np.zeros((self.K, self.K))
        
        for function in functions:
            for pos, token in enumerate(function):
                dist[pos, self.token_idx_map[token]] += 1
        
        # Normalize by number of permutations
        dist = dist / len(functions)
        return dist
    
    def calculate_pairwise_distribution(self, functions: List) -> np.ndarray:
        """
        Calculate P(token j follows token i) for all ordered pairs.
        Returns: K x K matrix where entry [i,j] = P(token j+1 follows token i+1)
        """
        dist = np.zeros((self.K, self.K))
        
        for function in functions:
            for i in range(len(function) - 1):
                token_i = self.token_idx_map[function[i]]
                token_j = self.token_idx_map[function[i+1]]
                dist[token_i, token_j] += 1
        
        # Normalize by total number of adjacencies
        total_adjacencies = len(functions) * (self.K - 1)
        dist = dist / total_adjacencies
        return dist

    def calculate_pairwise_distribution_position_wise(self, functions: List) -> np.ndarray:
        """
        Calculate P(token j follows token i) for all ordered pairs.
        Returns: K x K matrix where entry [i,j] = P(token j+1 follows token i+1)
        """
        dist = np.zeros((self.K, self.K, self.K))
        
        for function in functions:
            for i in range(len(function) - 1):
                token_i = self.token_idx_map[function[i]]
                token_j = self.token_idx_map[function[i+1]]
                dist[i, token_i, token_j] += 1
        
        # Normalize by total number of adjacencies
        total_adjacencies = len(functions)
        dist = dist / total_adjacencies
        return dist
    
    
    def kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate KL divergence D_KL(P || Q).
        Add small epsilon to avoid log(0).
        """
        # with a tolerance of 1e-6
        print(np.sum(p), np.sum(q))
        assert np.abs(np.sum(p) - 1) < 1e-6, "p must sum to 1"
        assert np.abs(np.sum(q) - 1) < 1e-6, "q must sum to 1"
        p = p + self.epsilon
        q = q + self.epsilon
        # Renormalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log(p / q))

    def position_coverage_divergence_kl(self, train_dist: np.ndarray, 
                                       test_dist: np.ndarray) -> float:
        """
        Calculate Position Coverage Divergence using KL divergence.
        Average KL divergence across all tokens.
        """
        kl_sum = 0
        for i in range(self.K):
            kl_sum += self.kl_divergence(test_dist[i, :], train_dist[i, :])
        
        return kl_sum / self.K

    def pairwise_adjacency_divergence_kl(self, train_dist: np.ndarray, 
                                       test_dist: np.ndarray) -> float:
        """
        Calculate Pairwise Adjacency Divergence using KL divergence.
        Average KL divergence across all tokens.
        """
        # kl_sum = 0
        # for i in range(self.K - 1):
            
                # train_flattened = train_dist[i, :].flatten()
                # test_flattened = test_dist[i, :].flatten()
        return self.kl_divergence(test_dist.flatten(), train_dist.flatten())
        # return kl_sum / self.K
        
    def combined_divergence_kl(self, pcd: float, pad: float, alpha: float = 0.5) -> float:
        """
        Combine PCD and PAD with weighting parameter alpha.
        """
        return alpha * pcd + (1 - alpha) * pad
    
    def analyze_split(self, train: List, test: List, unique_tokens: List[int]):
        """
        Comprehensive analysis of a train/test split.
        """
        print(f"\n{'='*60}")
        print(f"Train/Test Analysis")
        print(f"{'='*60}")
        print(f"Train size: {len(train)}, Test size: {len(test)}")
        
        # Calculate distributions
        train_pos_dist = self.calculate_position_distribution(train)
        test_pos_dist = self.calculate_position_distribution(test)
        
        train_pair_dist = self.calculate_pairwise_distribution(train)
        test_pair_dist = self.calculate_pairwise_distribution(test)
        # train_pair_dist = self.calculate_pairwise_distribution_position_wise(train)
        # test_pair_dist = self.calculate_pairwise_distribution_position_wise(test)
        
        # Calculate metrics
       
        pcd_kl = self.position_coverage_divergence_kl(train_pos_dist, test_pos_dist)
        pad_kl = self.pairwise_adjacency_divergence_kl(train_pair_dist, test_pair_dist)
        combined_kl = self.combined_divergence_kl(pcd_kl, pad_kl)
        
        print(f"\nMetrics:")
        print(f"  Position Coverage Divergence (KL): {pcd_kl:.6f}")
        print(f"  Pairwise Adjacency Divergence (KL): {pad_kl:.6f}")
        print(f"  Combined Divergence (KL) (α=0.5): {combined_kl:.6f}")
        # Show some distribution examples
        for i in range(self.K):
            print(f"\nPosition Distribution for position {i+1} (train vs test):")
            print(f"  Train: {train_pos_dist[i, :]}")
            print(f"  Test:  {test_pos_dist[i, :]}")
            print(f"  Diff:  {np.abs(train_pos_dist[i, :] - test_pos_dist[i, :])}")
            
        # print(f"\nPairwise Distribution for train vs test:")
        # print(f"  Train: {train_pair_dist.flatten()}")
        # print(f"  Test:  {test_pair_dist.flatten()}")
        # print(f"  Diff:  {np.abs(train_pair_dist.flatten() - test_pair_dist.flatten())}")
        # # find pair with max negative difference
        # max_diff = np.max(test_pair_dist.flatten() - train_pair_dist.flatten())
        # max_diff_index = np.argmax(test_pair_dist.flatten() - train_pair_dist.flatten())
        # print(f"  Max difference: {max_diff} at index {max_diff_index}")
        # # find module pair for max difference
        # module_pair = (max_diff_index // self.K, max_diff_index % self.K)
        # print(f"  Module pair: {module_pair}")
        # print(f"  Module: {unique_tokens[module_pair[0]]} -> {unique_tokens[module_pair[1]]}")
        return {
            'pcd_kl': pcd_kl,
            'pad_kl': pad_kl,
            'combined_kl': combined_kl,
            'train_pos_dist': train_pos_dist,
            'test_pos_dist': test_pos_dist,
            'train_pair_dist': train_pair_dist,
            'test_pair_dist': test_pair_dist
        }

    def get_test_accuracy(self, function_type: str, split_strategy: str, prompt_mode: str, pos_embedding_type: str) -> str:
        test_accuracies = []
        for seed in [0, 10, 20, 30, 40]:
            acc_path = f"{RESULTS_PATH}{function_type}/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_6/{prompt_mode}/train_{split_strategy}/eval_{split_strategy}/{pos_embedding_type}/nh6_nl3/seed_{seed}/accs.pkl"
            with open(acc_path, 'rb') as f:
                metrics = pickle.load(f)
                
            test_accuracies.append(metrics['test'].sharp_accuracy)
        return np.mean(test_accuracies), np.std(test_accuracies)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_max_length",
        type=int,
        default=6,
        help="compositional task max length"
    )
    parser.add_argument(
        "--function_type", type=str, default="uniform", help="uniform or diverse"
    )
    args = parser.parse_args()
    cfg = read_config("./config/gen/conf.yaml")
    cfg.task_max_length = args.task_max_length
    cfg.function_type = args.function_type

    # Split strategy values to try (permutation_6_{X})
    split_sizes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for epsilon in [1e-10]:
        results = {}
        systematic_metrics = {
            prompt_mode: {
                function_type: {
                    pos_embedding_type: []
                    for pos_embedding_type in ['abs', 'rel_global']
                }
                for function_type in ['uniform']
            }
            for prompt_mode in ['step_by_step']
        }

        random_metrics = {
            prompt_mode: {
                function_type: {
                    pos_embedding_type: []
                    for pos_embedding_type in ['abs', 'rel_global']
                }
                for function_type in ['uniform']
            }
            for prompt_mode in ['step_by_step']
        }
        for prompt_mode in ['step_by_step']:
            for function_type in ['uniform']:
                results[f"{prompt_mode}_{function_type}"] = {}
                for pos_embedding_type in ['abs', 'rel_global']:
                    results[f"{prompt_mode}_{function_type}"][pos_embedding_type] = {}
                    for strategy_sufix in ["continuouscoverage"]:
                        
                        results[f"{prompt_mode}_{function_type}"][pos_embedding_type][strategy_sufix] = []

                        for s in split_sizes:
                            if strategy_sufix == "random":
                                split_strategy = f"permutationrandom_6_{s}"
                            elif strategy_sufix == "systematic":
                                split_strategy = f"permutation_6_{s}"
                            elif strategy_sufix == "coverage":
                                split_strategy = f"coverage_6_{s}"
                            elif strategy_sufix == "reversecoverage":
                                split_strategy = f"reversecoverage_6_{s}"
                            elif strategy_sufix == "paircoverage":
                                split_strategy = f"paircoverage_6_{s}"
                            elif strategy_sufix == "reversepaircoverage":
                                split_strategy = f"reversepaircoverage_6_{s}"
                            elif strategy_sufix == "continuouscoverage":
                                split_strategy = f"continuouscoverage_6_{s}"
                            cfg.split_strategy = split_strategy
                            compositions_generator = CompositionsGenerator(cfg)
                            train_functions, test_functions, functions_info = compositions_generator.get_train_test_compositions()
                            print(f"Split strategy: {split_strategy}")
                            print("Length of train functions: ", len(train_functions))
                            print("Length of test functions: ", len(test_functions))
                            # find unique tokens in train and test functions
                            unique_tokens = train_functions[0]
                            print("Unique tokens: ", unique_tokens)
                            module_coverage_metric_calculation = ModuleCoverageMetricCalculator(cfg.task_max_length, unique_tokens, epsilon)
                            metrics = module_coverage_metric_calculation.analyze_split(train_functions, test_functions, unique_tokens)
                            divergence = metrics['combined_kl']
                            pairwise_divergence = metrics['pad_kl']
                            position_divergence = metrics['pcd_kl']
                            if strategy_sufix == "systematic":
                                systematic_metrics[prompt_mode][function_type][pos_embedding_type].append((s, divergence, pairwise_divergence, position_divergence))
                            else:
                                random_metrics[prompt_mode][function_type][pos_embedding_type].append((s, divergence, pairwise_divergence, position_divergence))
                            # mean_test_accuracy, std_test_accuracy = module_coverage_metric_calculation.get_test_accuracy(function_type, split_strategy, prompt_mode, pos_embedding_type)
                            mean_test_accuracy = 0
                            std_test_accuracy = 0
                            print(f"Strategy: {split_strategy}, Prompt mode: {prompt_mode}, Function type: {function_type}, Pos embedding type: {pos_embedding_type}, Divergence: {divergence}, Mean test accuracy: {mean_test_accuracy}, Std test accuracy: {std_test_accuracy}")
                            results[f"{prompt_mode}_{function_type}"][pos_embedding_type][strategy_sufix].append((divergence, mean_test_accuracy, std_test_accuracy))
        with open(f'reverse_pair_coverage_metrics_{epsilon}.pkl', 'wb') as f:
            pickle.dump(results, f)
        # save as pkl
        # with open(f'systematic_metrics_{epsilon}.pkl', 'wb') as f:
        #     pickle.dump(systematic_metrics, f)
        # with open(f'random_metrics_{epsilon}.pkl', 'wb') as f:
        #     pickle.dump(random_metrics, f)
        
        # plot divergence vs accuracy have systematic and random in the same plot
        # have pos embedding type in the same plot
        # for key, value in results.items():
        #     plt.figure(figsize=(7,5))
        #     # have scatter plot with error bars on marker (no lines)
        #     # Plot only the points and add error bars (with no connecting lines)
        #     for pos_embedding_type in value.keys():
        #         plt.figure(figsize=(7,5))
        #         plt.errorbar(
        #             [x[0] for x in value[pos_embedding_type]['random']],
        #             [x[1] for x in value[pos_embedding_type]['random']],
        #             yerr=[x[2] for x in value[pos_embedding_type]['random']],
        #             fmt='o',
        #             label='random',
        #             linestyle='None'
        #         )
        #         plt.errorbar(
        #             [x[0] for x in value[pos_embedding_type]['systematic']],
        #             [x[1] for x in value[pos_embedding_type]['systematic']],
        #             yerr=[x[2] for x in value[pos_embedding_type]['systematic']],
        #             fmt='o',
        #             label='systematic',
        #             linestyle='None'
        #         )
        #         plt.xlabel("Divergence")
        #         plt.ylabel("Accuracy")
        #         plt.title(f"{key} - {pos_embedding_type}")
        #         plt.legend(loc='best')
        #         plt.grid(True)
        #         plt.tight_layout()
        #         plt.savefig(f"module_coverage_metric_calculation_{key}_{pos_embedding_type}.png")
        #         plt.close()
        #     plt.xlabel("Divergence")
        #     plt.ylabel("Accuracy")
        #     plt.title(f"{key}")
        #     plt.legend(loc='best')
        #     plt.grid(True)
        #     plt.tight_layout()
        #     plt.savefig(f"module_coverage_metric_calculation_{key}.png")