"""
Split strategy classes for function train/test splitting.

This module contains different strategies for splitting function combinations
into training and test sets, including random, systematic, equivalence-based,
and custom strategies.
"""

import itertools
import json
from src.data.equivalence_classes.composition_equivalence_classes import CompositionEquivalenceClasses
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import os
import pickle
from init import ROOT_DIR
TRAIN_TEST_RATIO = 0.8

class BaseSplitStrategy(ABC):
    """Base class for all split strategies."""

    @abstractmethod
    def split(
        self, all_functions: List[List[str]], cfg
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Split functions into train and test sets.

        Args:
            all_functions: List of all function combinations
            cfg: Configuration object

        Returns:
            Tuple of (train_functions, test_functions)
        """
        pass


class RandomSplitStrategy(BaseSplitStrategy):
    """Random splitting strategy."""

    def split(
        self, all_functions: List[List[str]], cfg
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split functions randomly with train/test ratio."""
        shuffled_functions = all_functions.copy()
        random.shuffle(shuffled_functions)
        split_index = int(len(shuffled_functions) * TRAIN_TEST_RATIO)
        train_functions = shuffled_functions[:split_index]
        test_functions = shuffled_functions[split_index:]
        return train_functions, test_functions


class CombinationRandomSplitStrategy(BaseSplitStrategy):
    """Random splitting for combinations with fixed or variable prompt length."""

    def split(
        self, all_functions: List[List[str]], cfg
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split combinations randomly."""
        K = int(cfg.split_strategy.split("_")[1])
        if cfg.prompt_length == "fixed":
            return self._split_fixed(all_functions, K)
        else:
            return self._split_variable(all_functions, K)

    def _split_fixed(
        self, all_functions: List[List[str]], K: int
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split fixed-length combinations."""
        K_max = 6
        target_K = K if K <= K_max else K_max

        num_i_functions = [
            fn_list
            for fn_list in all_functions
            if len(fn_list) - fn_list.count("identity") == target_K
        ]

        train_functions = random.sample(
            num_i_functions, int(len(num_i_functions) * TRAIN_TEST_RATIO)
        )
        test_functions = [fn for fn in num_i_functions if fn not in train_functions]

        print(f"Train functions: {len(train_functions)}")
        print(f"Test functions: {len(test_functions)}")
        return train_functions, test_functions

    def _split_variable(
        self, all_functions: List[List[str]], K: int
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split variable-length combinations."""
        num_i_functions = [
            fn_list
            for fn_list in all_functions
            if len(fn_list) - fn_list.count("identity") == K
        ]

        if K == 1:
            return num_i_functions, num_i_functions

        train_functions = random.sample(
            num_i_functions, int(len(num_i_functions) * TRAIN_TEST_RATIO)
        )
        test_functions = [fn for fn in num_i_functions if fn not in train_functions]

        print(f"Train functions: {len(train_functions)}")
        print(f"Test functions: {len(test_functions)}")
        return train_functions, test_functions


class PermutationSplitStrategy(BaseSplitStrategy):
    """Systematic permutation-based splitting."""

    def __init__(self, cfg, function_names: List[str]):
        self.function_names = function_names
        self.strategy = cfg.split_strategy
        self.cfg = cfg
        self._init_strategy_params()

    def _init_strategy_params(self):
        """Initialize strategy-specific parameters."""
        split_list = self.strategy.split("_")
        if len(split_list) == 4:
            self.multiple_relative_order = True
            self.num_relative_order = int(split_list[2])
            self.equivalence_class_leakage = int(split_list[3])
        elif len(split_list) == 3:
            self.multiple_relative_order = True
            self.num_relative_order = int(split_list[2])
        else:
            self.multiple_relative_order = False
            self.num_relative_order = 1

    def split(
        self, all_functions: List[List[str]], cfg
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split using systematic permutation strategy."""
        if cfg.prompt_length == "fixed":
            return self._split_fixed(all_functions)
        else:
            return self._split_variable(all_functions)

    def _get_relative_order(self) -> List[str]:
        """Get the relative order of the functions."""
        if self.multiple_relative_order:
            return self._generate_multiple_relative_order()
        else:
            return [self._generate_all_relative_order()[0]]

    def _generate_all_relative_order(self) -> List[str]:
        """Generate all possible relative order permutations of the functions."""
        function_names_without_identity = [
            function for function in self.function_names if function != "identity"
        ]
        return list(
            itertools.permutations(
                function_names_without_identity, len(function_names_without_identity)
            )
        )

    def _generate_multiple_relative_order(self) -> List[str]:
        """Generate the relative order of the functions."""
        all_relative_order = self._generate_all_relative_order()

        if self.strategy.startswith("permutationrandom"):
            np.random.seed(self.cfg.seed)
            if self.strategy == "permutationrandom_6_1":
                np.random.shuffle(all_relative_order)
                with open(
                    f"{ROOT_DIR}/data/jsons/relative_order_{self.strategy}_{self.cfg.function_type}.json",
                    "w",
                ) as f:
                    json.dump(all_relative_order, f)
            else:
                with open(
                    f"{ROOT_DIR}/data/jsons/relative_order_permutationrandom_6_1_{self.cfg.function_type}.json",
                    "r",
                ) as f:
                    all_relative_order = json.load(f)
            return all_relative_order[: self.num_relative_order]

        elif self.strategy.startswith("permutationrotated"):
            with open(
                f"{ROOT_DIR}/data/jsons/relative_order_permutationrandom_6_1_{self.cfg.function_type}.json",
                "r",
            ) as f:
                all_relative_order = json.load(f)

            first_six = all_relative_order[:6]
            rotations = first_six.copy()
            for j in range(len(all_relative_order[0]) - 1):
                for i in range(len(first_six)):
                    new_rotation = first_six[i][j + 1 :] + first_six[i][: j + 1]
                    rotations.append(new_rotation)
            return rotations[: self.num_relative_order]

        elif self.strategy.startswith("permutationcoveragev1"):
            function_names_without_identity = [
                function
                for function in self.function_names
                if function != "identity"
            ]
            num_orderings = int(self.strategy.split("_")[2])
            print("num_orderings", num_orderings)
            with open(
                f"{ROOT_DIR}/data/csvs/absolute_coverage_permutations_30.csv", "r"
            ) as f:
                relative_order_index_list = pd.read_csv(f)
            relative_order_index_list = relative_order_index_list.values.tolist()
            relative_order_index_list = relative_order_index_list[:num_orderings]
            function_lists = []
            for i in range(len(relative_order_index_list)):
                function_list = [
                    function_names_without_identity[i - 1]
                    for i in relative_order_index_list[i]
                ]
                function_lists.append(function_list)
            return function_lists

        elif (
            self.strategy.startswith("permutation")
            or self.strategy.startswith("equivalence")
            or self.strategy.startswith("uniequivalence")
        ):
            return all_relative_order[: self.num_relative_order]
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _split_fixed(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split fixed-length permutations systematically."""
        function_names_without_identity = [
            function for function in self.function_names if function != "identity"
        ]
        relative_order_list = self._get_relative_order()

        K = int(self.cfg.split_strategy.split("_")[1])
        K_max = 6
        target_K = K if K <= K_max else K_max

        i_functions = [
            fn_list
            for fn_list in all_functions
            if len(fn_list) - fn_list.count("identity") == target_K
        ]

        selected_combinations = []
        unseen_combinations = []

        if K != 1:
            for i in range(len(i_functions)):
                combo = i_functions[i]
                combo = [function for function in combo if function != "identity"]
                given_order_combo = list(combo)

                select_combo = False
                for relative_order in relative_order_list:
                    filtered_combo = [fn for fn in relative_order if fn in combo]
                    if filtered_combo == given_order_combo:
                        selected_combinations.append(i_functions[i])
                        select_combo = True
                        break

                if not select_combo:
                    unseen_combinations.append(i_functions[i])
        else:
            for i in range(len(i_functions)):
                combo = i_functions[i]
                non_identity_index = combo.index(
                    next(function for function in combo if function != "identity")
                )
                non_identity_function = combo[non_identity_index]

                select_combo = False
                for relative_order in relative_order_list:
                    same_function_index = relative_order.index(non_identity_function)
                    same_function_index_in_combo = combo.index(
                        relative_order[same_function_index]
                    )
                    if same_function_index_in_combo == same_function_index:
                        selected_combinations.append(i_functions[i])
                        select_combo = True
                        break

                if not select_combo:
                    unseen_combinations.append(i_functions[i])

        if len(selected_combinations) == 0 or len(unseen_combinations) == 0:
            print(f"No selected or unseen combinations found for size {K}")
            unseen_combinations = i_functions
            selected_combinations = i_functions

        print(f"Selected combinations: {len(selected_combinations)}")
        print(f"Unseen combinations: {len(unseen_combinations)}")
        return selected_combinations, unseen_combinations

    def _split_variable(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split variable-length permutations systematically."""
        relative_order_list = self._get_relative_order()

        K = int(self.cfg.split_strategy.split("_")[1])
        K_max = 6
        target_K = K if K <= K_max else K_max

        i_functions = [
            fn_list for fn_list in all_functions if len(fn_list) == target_K
        ]

        selected_combinations = []
        unseen_combinations = []

        for i in range(len(i_functions)):
            combo = i_functions[i]
            select_combo = False
            for relative_order in relative_order_list:
                selected_relative_order = []
                for j in relative_order:
                    if j in combo:
                        selected_relative_order.append(j)
                if selected_relative_order == combo:
                    selected_combinations.append(combo)
                    select_combo = True
                    break

            if not select_combo:
                unseen_combinations.append(combo)

        if len(selected_combinations) == 0 or len(unseen_combinations) == 0:
            print(f"No selected or unseen combinations found for size {K}")
            unseen_combinations = i_functions
            selected_combinations = i_functions

        return selected_combinations, unseen_combinations


class EquivalenceSplitStrategy(BaseSplitStrategy):
    """Equivalence class-based splitting."""

    def __init__(self, cfg, function_names: List[str]):
        self.function_names = function_names
        self.strategy = cfg.split_strategy
        self.cfg = cfg
        self.equivalence_class_leakage = self._extract_leakage()

    def _extract_leakage(self) -> int:
        """Extract equivalence class leakage from strategy."""
        split_list = self.strategy.split("_")
        if len(split_list) == 4:
            return int(split_list[3])
        else:
            return 0

    def split(
        self, all_functions: List[List[str]], cfg
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split using equivalence class strategy."""
        if cfg.prompt_length == "fixed":
            return self._split_fixed(all_functions)
        else:
            return self._split_variable(all_functions)

    def _learn_equivalence_class_map(
        self, function_lists: List[List[str]]
    ) -> Tuple[Dict[Tuple[str, ...], List[int]], List[Tuple[str, ...]]]:
        """Learn equivalence class map from function lists."""
        equivalence_class_map = {}
        unique_function_lists = []
        for i, function_list in enumerate(function_lists):
            function_list = [
                function for function in function_list if function != "identity"
            ]
            equivalence_class = tuple(function_list)
            if equivalence_class not in equivalence_class_map:
                equivalence_class_map[equivalence_class] = []
                unique_function_lists.append(equivalence_class)
            equivalence_class_map[equivalence_class].append(i)
        return equivalence_class_map, unique_function_lists

    def _split_fixed(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split fixed-length equivalence classes."""
        # First get systematic split
        permutation_strategy = PermutationSplitStrategy(
            self.cfg, self.function_names
        )
        train_functions, test_functions = permutation_strategy._split_fixed(all_functions)

        train_equivalence_class_map, train_unique_function_lists = (
            self._learn_equivalence_class_map(train_functions)
        )
        test_equivalence_class_map, test_unique_function_lists = (
            self._learn_equivalence_class_map(test_functions)
        )

        if self.equivalence_class_leakage == 0:
            return train_functions, test_functions

        # Apply leakage
        all_train_indices = list(range(len(train_functions)))
        all_test_indices = list(range(len(test_functions)))

        final_train_functions = []
        final_test_functions = []
        sampled_test_indices = []
        sampled_train_indices = []

        for i in range(self.equivalence_class_leakage):
            candidate_train_equivalence_class = train_unique_function_lists[i]
            candidate_test_equivalence_class = test_unique_function_lists[i]

            train_functions_indices = train_equivalence_class_map[
                candidate_train_equivalence_class
            ]
            test_functions_indices = test_equivalence_class_map[
                candidate_test_equivalence_class
            ]

            test_functions_indices_sampled = random.sample(
                test_functions_indices, int(0.5 * len(test_functions_indices))
            )
            train_functions_indices_sampled = random.sample(
                test_functions_indices, int(0.5 * len(test_functions_indices))
            )

            sampled_test_indices.extend(test_functions_indices_sampled)
            sampled_train_indices.extend(train_functions_indices_sampled)

        remaining_test_functions_indices = [
            i for i in all_test_indices if i not in sampled_test_indices
        ]
        remaining_train_functions_indices = [
            i for i in all_train_indices if i not in sampled_train_indices
        ]

        for i in sampled_test_indices:
            final_train_functions.append(test_functions[i])
        for i in sampled_train_indices:
            final_test_functions.append(train_functions[i])
        for i in remaining_train_functions_indices:
            final_train_functions.append(train_functions[i])
        for i in remaining_test_functions_indices:
            final_test_functions.append(test_functions[i])

        return final_train_functions, final_test_functions

    def _split_variable(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split variable-length equivalence classes."""
        permutation_strategy = PermutationSplitStrategy(
            self.cfg, self.function_names
        )
        return permutation_strategy._split_variable(all_functions)


class UniqueEquivalenceSplitStrategy(BaseSplitStrategy):
    """Unique equivalence class-based splitting."""

    def __init__(self, cfg, function_names: List[str]):
        self.function_names = function_names
        self.strategy = cfg.split_strategy
        self.cfg = cfg
        split_list = self.strategy.split("_")
        if len(split_list) == 4:
            self.equivalence_class_leakage = int(split_list[3])
        else:
            self.equivalence_class_leakage = 0

    def split(
        self, all_functions: List[List[str]], cfg
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split using unique equivalence class strategy."""
        if cfg.prompt_length == "fixed":
            return self._split_fixed(all_functions)
        else:
            return None, None

    def _learn_equivalence_class_map(
        self, function_lists: List[List[str]]
    ) -> Tuple[Dict[Tuple[str, ...], List[int]], List[Tuple[str, ...]]]:
        """Learn equivalence class map from function lists."""
        equivalence_class_map = {}
        unique_function_lists = []
        for i, function_list in enumerate(function_lists):
            function_list = [
                function for function in function_list if function != "identity"
            ]
            equivalence_class = tuple(function_list)
            if equivalence_class not in equivalence_class_map:
                equivalence_class_map[equivalence_class] = []
                unique_function_lists.append(equivalence_class)
            equivalence_class_map[equivalence_class].append(i)
        return equivalence_class_map, unique_function_lists

    def _split_fixed(
        self, all_functions: List[List[str]]
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split fixed-length unique equivalence classes."""
        permutation_strategy = PermutationSplitStrategy(
            self.cfg, self.function_names
        )
        train_functions, test_functions = permutation_strategy._split_fixed(all_functions)

        all_functions = train_functions + test_functions
        equivalence_class_map, unique_function_lists = self._learn_equivalence_class_map(
            all_functions
        )

        new_train_function_indices = []
        new_test_function_indices = []

        for unique_function_list in unique_function_lists:
            equivalence_class_functions_indices = equivalence_class_map[unique_function_list]
            train_function_index = random.sample(
                equivalence_class_functions_indices,
                self.equivalence_class_leakage + 1,
            )
            new_train_function_indices.extend(train_function_index)

            remaining_test_functions_indices = [
                i
                for i in equivalence_class_functions_indices
                if i not in train_function_index
            ]
            new_test_function_indices.extend(remaining_test_functions_indices)

        new_train_function_indices = list(set(new_train_function_indices))
        new_test_function_indices = list(set(new_test_function_indices))

        new_train_functions = [all_functions[i] for i in new_train_function_indices]
        new_test_functions = [all_functions[i] for i in new_test_function_indices]

        print(f"Train indices: {len(new_train_function_indices)}")
        print(f"Test indices: {len(new_test_function_indices)}")

        return new_train_functions, new_test_functions


class CustomSplitStrategy(BaseSplitStrategy):
    """Custom splitting strategy."""

    def split(
        self, all_functions: List[List[str]], cfg
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split using custom strategy."""
        selected_functions = []
        for fn_list in all_functions:
            selected = True
            for fn in fn_list:
                if fn not in ["sort", "max", "filter", "map"] or len(fn_list) - fn_list.count("identity") != 3:
                    selected = False
                    break
            if selected:
                selected_functions.append(fn_list)

        train_functions = [
            ["sort", "max", "filter"],
            ["filter", "sort", "max"],
            ["max", "sort", "filter"],
            ["sort", "filter", "max"]
        ]
        test_functions = [fn for fn in selected_functions if fn not in train_functions]

        return train_functions, test_functions

class DisJointSplitStrategy(BaseSplitStrategy):
    """Disjoint splitting strategy."""

    def __init__(self, cfg, function_names: List[str]):
        self.MODE = "odd"
        self.function_names = function_names
        self.cfg = cfg
        
    def split(
        self, all_functions: List[List[str]], cfg
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Split using disjoint strategy."""
        K = int(self.cfg.split_strategy.split("_")[1])
        
        
        strategy_prefix = cfg.split_strategy.split("_")[0]
        shared_equivalence_classes_percentage = int(self.cfg.split_strategy.split("_")[2])
        shared_eq_float = round(float(shared_equivalence_classes_percentage)/100, 1)
        if shared_equivalence_classes_percentage == 0:
            # round to 1 decimal place
            composition_equivalence_classes = CompositionEquivalenceClasses(K)
            composition_equivalence_classes.load_ce_metric()
            composition_equivalence_classes.convert_ce_metric_to_matrix()
            composition_equivalence_classes.cluster_ce_metric_matrix()
            composition_equivalence_classes.print_cluster_info()
            train_task_indices, test_task_indices = composition_equivalence_classes.splitter.create_disjoint_splits(
                seed=5, train_cluster_threshold=4, mode=self.MODE)
            four_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            six_ratios = [0.0, 0.1, 0.2, 0.4, 0.6, 0.7, 1.0]
            if K == 4:
                ratios = four_ratios
            elif K == 6:
                ratios = six_ratios
            for percentage in ratios:
                train_functions, test_functions, number_leaked_test_tasks = composition_equivalence_classes.split_train_test_functions(
                    train_task_indices, test_task_indices, seed=5, shared_equivalence_classes_percentage=percentage, mode=self.MODE)
                print(f"Percentage: {percentage}")
                print(f"Number of leaked test tasks: {number_leaked_test_tasks}/{len(test_functions)} = {number_leaked_test_tasks / len(test_functions)}")

                # save train and test functions to pkl file

                dir_path = f"{ROOT_DIR}/data/equivalence_classes/{strategy_prefix}/{K}/{percentage}"
                os.makedirs(dir_path, exist_ok=True)
                with open(f"{dir_path}/train_functions.pkl", "wb") as f:
                    pickle.dump(train_functions, f)
                with open(f"{dir_path}/test_functions.pkl", "wb") as f:
                    pickle.dump(test_functions, f)
                # save number of leaked test tasks to txt file
                number_leaked_test_tasks = number_leaked_test_tasks / len(test_functions)
                with open(f"{dir_path}/number_leaked_test_tasks.json", "w") as f:
                    json.dump({"number_leaked_test_tasks": number_leaked_test_tasks}, f)
        dir_path = f"{ROOT_DIR}/data/equivalence_classes/{strategy_prefix}/{K}/{shared_eq_float}"
        with open(f"{dir_path}/train_functions.pkl", "rb") as f:
            train_functions = pickle.load(f)
        with open(f"{dir_path}/test_functions.pkl", "rb") as f:
            test_functions = pickle.load(f)
        return train_functions, test_functions

class DisJointSplitStrategy2(DisJointSplitStrategy):
    """Disjoint splitting strategy (even mode)."""

    def __init__(self, cfg, function_names: List[str]):
        super().__init__(cfg, function_names)
        self.MODE = "even"
        self.cfg = cfg
        self.function_names = function_names
class DisJointSplitStrategy3(DisJointSplitStrategy):
    """Disjoint splitting strategy (even mode)."""

    def __init__(self, cfg, function_names: List[str]):
        super().__init__(cfg, function_names)
        self.MODE = "odd_reverse"
        self.cfg = cfg
        self.function_names = function_names


class DisJointSplitStrategy4(DisJointSplitStrategy):
    """Disjoint splitting strategy (even mode)."""

    def __init__(self, cfg, function_names: List[str]):
        super().__init__(cfg, function_names)
        self.MODE = "even_reverse"
        self.cfg = cfg
        self.function_names = function_names

class DisJointSplitStrategy5(DisJointSplitStrategy):
    """Disjoint splitting strategy (even mode)."""

    def __init__(self, cfg, function_names: List[str]):
        super().__init__(cfg, function_names)
        self.MODE = "odd_exchange"
        self.cfg = cfg
        self.function_names = function_names

class DisJointSplitStrategy6(DisJointSplitStrategy):
    """Disjoint splitting strategy (even mode)."""

    def __init__(self, cfg, function_names: List[str]):
        super().__init__(cfg, function_names)
        self.MODE = "even_exchange"
        self.cfg = cfg
        self.function_names = function_names

class DisJointSplitStrategy7(DisJointSplitStrategy):
    """Disjoint splitting strategy (even mode)."""

    def __init__(self, cfg, function_names: List[str]):
        super().__init__(cfg, function_names)
        self.MODE = "odd_exchange_reverse"
        self.cfg = cfg
        self.function_names = function_names

class DisJointSplitStrategy8(DisJointSplitStrategy):    
    """Disjoint splitting strategy (even mode)."""

    def __init__(self, cfg, function_names: List[str]):
        super().__init__(cfg, function_names)
        self.MODE = "even_exchange_reverse"
        self.cfg = cfg
        self.function_names = function_names

class DisJointSplitStrategy9(DisJointSplitStrategy):
    """Disjoint splitting strategy (even mode)."""

    def __init__(self, cfg, function_names: List[str]):
        super().__init__(cfg, function_names)
        self.MODE = "even"
        self.cfg = cfg
        self.function_names = function_names

def get_split_strategy(cfg, function_names: List[str]) -> BaseSplitStrategy:
    """
    Factory function to get the appropriate split strategy.

    Args:
        cfg: Configuration object
        function_names: List of function names

    Returns:
        Appropriate split strategy instance
    """
    if cfg.split_strategy == "random":
        return RandomSplitStrategy()
    elif cfg.split_strategy.startswith("combination"):
        return CombinationRandomSplitStrategy()
    elif cfg.split_strategy.startswith("permutation"):
        return PermutationSplitStrategy(cfg, function_names)
    elif cfg.split_strategy.startswith("equivalence"):
        return EquivalenceSplitStrategy(cfg, function_names)
    elif cfg.split_strategy.startswith("uniequivalence"):   
        return UniqueEquivalenceSplitStrategy(cfg, function_names)
    elif cfg.split_strategy.startswith("custom"):
        return CustomSplitStrategy()
    elif cfg.split_strategy.startswith("disjoint3"):
        return DisJointSplitStrategy3(cfg, function_names)
    elif cfg.split_strategy.startswith("disjoint4"):
        return DisJointSplitStrategy4(cfg, function_names)
    elif cfg.split_strategy.startswith("disjoint2"):
        return DisJointSplitStrategy2(cfg, function_names)
    elif cfg.split_strategy.startswith("disjoint1"):
        return DisJointSplitStrategy(cfg, function_names)
    elif cfg.split_strategy.startswith("disjoint5"):
        return DisJointSplitStrategy5(cfg, function_names)
    elif cfg.split_strategy.startswith("disjoint6"):
        return DisJointSplitStrategy6(cfg, function_names)
    elif cfg.split_strategy.startswith("disjoint7"):
        return DisJointSplitStrategy7(cfg, function_names)
    elif cfg.split_strategy.startswith("disjoint8"):
        return DisJointSplitStrategy8(cfg, function_names)
    elif cfg.split_strategy.startswith("disjoint9"):
        return DisJointSplitStrategy9(cfg, function_names)
    else:   
        raise ValueError(f"Unknown split strategy: {cfg.split_strategy}")
