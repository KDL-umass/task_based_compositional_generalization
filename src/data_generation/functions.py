import functools
import itertools
import random
import string
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sympy.utilities.iterables import multiset_permutations

from src.data_generation.init import read_config
from src.data_generation.utils import *
from src.data_generation.split_strategies import get_split_strategy
from src.data_generation.function_combination_generator import FunctionCombinationGenerator

ROOT_DIR = "/project/pi_jensen_umass_edu/ppruthi_umass_edu/task_based_compositional_generalization"
TRAIN_TEST_RATIO = 0.8
import json


class BaseFunction:
    @staticmethod
    def identity(xstr):
        # max length of the string is 12
        return xstr

    @staticmethod
    def map(xstr, offset=1, n_alphabets=26):
        """
        Maps a string to a new string by replacing each character with its corresponding character in the alphabet.
        """
        # make sure charaacter is in a-chr(n_alphabets)
        return "".join(
            chr((ord(c) - ord("a") + offset) % n_alphabets + ord("a")) for c in xstr
        )

    @staticmethod
    def sort(xstr):
        """
        Sorts the characters in a string.
        """
        # max length of the string is 12
        return "".join(sorted(xstr))

    @staticmethod
    def aggregate(xstr):
        # returns the sum of the ASCII values of the characters in the string and converts it to a string by modulo 256
        # makes sure the result is a string in a-z
        return "".join(chr((sum(ord(c) for c in xstr) % 26) + ord("a")))

    @staticmethod
    def join(xstr1, xstr2):
        """
        Joins two strings together.
        """
        # max length of the string is 12
        return xstr1 + xstr2

    @staticmethod
    def union(xstr1, xstr2):
        """
        Returns the intersection of two strings.
        """
        # maintain the order of the first and second string
        # keeps the first string and adds the characters from the second string that are not in the first string

        ans = []
        for c in xstr1:
            if c not in ans:
                ans.append(c)
        for c in xstr2:
            if c not in ans:
                ans.append(c)
        return "".join(ans)

    @staticmethod
    def max(xstr1):
        """
        Finds the letter with the maximum occurrences in the string.
        If multiple characters have the same occurrence, chooses the alphabetically smallest one.
        """
        if len(xstr1) == 0:
            return ""
        from collections import Counter

        # Count occurrences of each character
        char_count = Counter(xstr1)

        # Find the maximum occurrence count
        max_count = max(char_count.values())

        # Filter characters with the maximum count and return the smallest alphabetically
        max_chars = [char for char, count in char_count.items() if count == max_count]
        return min(max_chars)

    def filter(xstr, filter_func=lambda x: x in "aeiou"):
        """
        Filters a string based on a filter function.
        """
        # max length of the string is 12
        return "".join(filter(filter_func, xstr))

class Diverse_v2_Function:
    @staticmethod
    def identity(xstr):
        # max length of the string is 12
        return xstr

    def map(xstr, offset=1, n_alphabets=26):
        mapping_json_file = f"{ROOT_DIR}/data/jsons/mappings.json"
        with open(mapping_json_file, "r") as f:
            mapping = json.load(f)
        return "".join(mapping["map1"][c] for c in xstr)

    @staticmethod
    def sort(xstr):
        """
        Sorts the characters in a string.
        """
        # max length of the string is 12
        return "".join(sorted(xstr))

    @staticmethod
    def join(xstr1, xstr2):
        """
        Joins two strings together.
        """
        # max length of the string is 12
        return xstr1 + xstr2

    @staticmethod
    def union(xstr1, xstr2):
        """
        Returns the intersection of two strings.
        """
        # maintain the order of the first and second string
        # keeps the first string and adds the characters from the second string that are not in the first string

        ans = []
        for c in xstr2:
            if c not in ans:
                ans.append(c)
        for c in xstr1:
            if c not in ans:
                ans.append(c)
        return "".join(ans)

    @staticmethod
    def max(xstr1):
        """
        Finds the letter with the maximum occurrences in the string.
        If multiple characters have the same occurrence, chooses the alphabetically smallest one.
        """
        # implement reverse sort
        return "".join(sorted(xstr1, reverse=True))

    def filter(xstr, filter_func):
        """
        Filters a string based on a filter function.
        """
        # filter consonants
        filter_func = lambda x: x not in "aeiou"
        return "".join(filter(filter_func, xstr))


class MapRandom:
    @staticmethod
    def map_random(seed, n_alphabets=26):
        """
        Returns a function that maps a string using a random mapping generated with the given seed.
        Also provides a way to get the mapping dictionary for inspection/storage.
        """
        np.random.seed(seed)
        random.seed(seed)
        # mapping is without replacement
        # sample n_alphabets characters from a-z
        sampled_characters = random.sample(list(string.ascii_lowercase), n_alphabets)
        mapping = {chr(i + ord("a")): sampled_characters[i] for i in range(n_alphabets)}

        def map_func(xstr):
            return "".join(mapping[c] for c in xstr)

        map_func.mapping = mapping  # Attach mapping dict for external access
        
        return map_func


DIVERSE_FUNCTIONS = {
    "sort": BaseFunction.sort,
    "join": BaseFunction.join,
    "filter": BaseFunction.filter,
    "map": BaseFunction.map,
    "union": BaseFunction.union,
    "max": BaseFunction.max,
    "identity": BaseFunction.identity,
}

DIVERSE_2_FUNCTIONS = {
    "sort": Diverse_v2_Function.sort,
    "join": Diverse_v2_Function.join,
    "filter": Diverse_v2_Function.filter,
    "map": Diverse_v2_Function.map,
    "union": Diverse_v2_Function.union,
    "max": Diverse_v2_Function.max,
    "identity": Diverse_v2_Function.identity,
}


class CreateFunctions:
    def __init__(self, cfg):
        self.n_alphabets = cfg.n_alphabets
        self.seq_len = cfg.seq_len
        self.function_properties = cfg.function
        self.n_functions = cfg.function.n_functions
        self.seed = cfg.seed
        self.function_type = cfg.function.type
        self.cfg = cfg
        
        # Initialize function dictionary based on type
        if cfg.function.type == "diverse":
            self.function_dict = DIVERSE_FUNCTIONS
            self.mappings = None
        elif cfg.function.type == "uniform":
            self.function_dict = {
                f"map{i}": MapRandom.map_random(seed=i)
                for i in range(1, self.n_functions + 1)
            }
            self.function_dict["identity"] = BaseFunction.identity
            self.mappings = {
                f"map{i}": MapRandom.map_random(seed=i).mapping
                for i in range(1, self.n_functions + 1)
            }
        elif cfg.function.type == "diverse2":
            self.function_dict = DIVERSE_2_FUNCTIONS
        
        self.strategy = self.function_properties.split.strategy
        self.function_names = list(self.function_dict.keys())
        
        # Initialize generators and strategies
        self._init_split_strategy()
        self._init_combination_generator()
        
        # Extract strategy parameters
        self._parse_strategy_params()

    def _init_split_strategy(self):
        """Initialize the split strategy based on configuration."""
        self.split_strategy = get_split_strategy(
            self.strategy, self.function_names, self.cfg
        )

    def _init_combination_generator(self):
        """Initialize the combination generator."""
        self.combination_generator = FunctionCombinationGenerator(self.cfg)
        self.combination_generator.set_function_names(self.function_names)

    def _parse_strategy_params(self):
        """Parse strategy parameters from the strategy string."""
        split_list = self.strategy.split("_")
        if len(split_list) >= 2:
            try:
                self.K = int(split_list[1])
            except (ValueError, IndexError):
                self.K = 1
        else:
            self.K = 1
            
        # Determine if multiple relative orders are needed
        if len(split_list) >= 3:
            self.multiple_relative_order = True
            try:
                self.num_relative_order = int(split_list[2])
            except (ValueError, IndexError):
                self.num_relative_order = 1
        else:
            self.multiple_relative_order = False
            self.num_relative_order = 1
            
        # Extract equivalence class leakage if present
        if len(split_list) >= 4:
            try:
                self.equivalence_class_leakage = int(split_list[3])
            except (ValueError, IndexError):
                self.equivalence_class_leakage = 0
        else:
            self.equivalence_class_leakage = 0

    def create_functions(self) -> Tuple[List[List[str]], Dict[Tuple[str, ...], int]]:
        """
        Create function combinations based on the specified strategy.

        Returns:
            Tuple of (all_functions, combination_ids)
        """
        # Generate combinations using the generator
        unique_function_combinations_permutations = self.combination_generator.generate(
            self.strategy
        )

        # Create combination IDs mapping
        combination_ids = {
            tuple(combo): i
            for i, combo in enumerate(unique_function_combinations_permutations)
        }
        # Convert to list format for consistency
        all_functions = [
            list(combo) for combo in unique_function_combinations_permutations
        ]

        self.combination_ids = combination_ids
        return all_functions, combination_ids

    def get_train_functions(
        self,
    ) -> Tuple[List[List[str]], List[List[str]], Dict[Tuple[str, ...], int]]:
        """
        Returns training and test function splits based on the configured strategy.

        Returns:
            Tuple of (train_functions, test_functions, functions_info)
        """
        all_functions, functions_info = self.create_functions()

        # Use the split strategy to divide functions into train and test
        train_functions, test_functions = self.split_strategy.split(
            all_functions, self.cfg
        )

        self._print_split_info(train_functions, test_functions)
        return train_functions, test_functions, functions_info

    def _print_split_info(
        self, train_functions: List[List[str]], test_functions: List[List[str]]
    ) -> None:
        """Print information about the train/test split."""
        print(f"Total number of training functions: {len(train_functions)}")
        print(f"Total number of test functions: {len(test_functions)}")


def apply_function_composition_diverse(
    function_list, function_dict, xstr1, xstr2, filter_func=lambda x: x in "aeiou", offset=1, n_alphabets=26
):
    """
    Applies a function to a string.
    """
    outputs = []
    # apply the function to the string
    for function in function_list:
        if function == "join" or function == "union":
            # if the function is join or intersect, apply it to both strings
            xstr1 = function_dict[function](xstr1, xstr2)
        elif function == "filter":
            xstr1 = function_dict[function](xstr1, filter_func=filter_func)
        elif function == "map":
            xstr1 = function_dict[function](xstr1, offset, n_alphabets)
        else:
            xstr1 = function_dict[function](xstr1)

        outputs.append(xstr1)
    return outputs


def apply_function_composition_uniform(function_list, function_dict, xstr1):
    """
    Applies a function to a string.
    """
    outputs = []
    # apply the function to the string
    for function in function_list:
        xstr1 = function_dict[function](xstr1)
        outputs.append(xstr1)
    return outputs

def apply_function_composition(function_type, function_list, function_dict, xstr1, xstr2=None, filter_func=lambda x: x in "aeiou", offset=1, n_alphabets=26):
    if function_type == "diverse":
        return apply_function_composition_diverse(function_list, function_dict, xstr1, xstr2, filter_func, offset, n_alphabets)
    elif function_type == "uniform":
        return apply_function_composition_uniform(function_list, function_dict, xstr1)
    else:
        raise ValueError(f"Invalid function type: {function_type}")


# test the functions
def main():
    cfg_path = "{}/config/gen/conf.yaml".format(ROOT_DIR)
    # read the config file
    cfg = read_config(cfg_path)
    cfg.prompt_length = "fixed"
    cfg.function.split.strategy = "combination_3"
    cfg.function.type = "diverse"
    cfg.task_max_length = 3
    # create the functions
    create_functions = CreateFunctions(cfg)
    all_functions, combination_ids = create_functions.create_functions()
    # function_lists = random.sample(all_functions, 20)
    train_functions, test_functions, functions_info = (
        create_functions.get_train_functions()
    )
    # print first 5 train functions
    for i in range(len(train_functions)):
        print(train_functions[i])
    # print first 5 test functions
    for i in range(5):
        print(test_functions[i])


    print("train_functions", len(train_functions))
    print("test_functions", len(test_functions))


if __name__ == "__main__":
    main()
