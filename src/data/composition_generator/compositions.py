from typing import Dict, List, Tuple
from init import ROOT_DIR, read_config
from src.data.modules.diverse import DIVERSE_FUNCTIONS, DIVERSE_2_FUNCTIONS, DiverseFunctionsV1
from src.data.modules.randomFunctions import MapRandom
from src.data.composition_generator.split_strategies import get_split_strategy
from src.data.composition_generator.function_combination_generator import FunctionCombinationGenerator

# This class is the base class for all compositions classes and it initializes the function dictionary based on the configuration.
# and applies the function composition based on the configuration.
class BaseCompositionsClass:
    def __init__(self, cfg):
        self.cfg = cfg
        self._init_function_dict()
    
    def _init_function_dict(self):
        """Initialize the function dictionary based on the configuration."""
        if self.cfg.composition.type == "diverse":
            self.function_dict = DIVERSE_FUNCTIONS
            self.mappings = None
        elif self.cfg.composition.type == "uniform":
            self.function_dict = {
                f"map{i}": MapRandom.map_random(seed=i).mapping
                for i in range(1, self.cfg.composition.n_functions + 1)
            }
            self.function_dict["identity"] = DiverseFunctionsV1.identity
            self.mappings = {
                f"map{i}": MapRandom.map_random(seed=i).mapping
                for i in range(1, self.cfg.composition.n_functions + 1)
            }
        elif self.cfg.composition.type == "diverse2":
            self.function_dict = DIVERSE_2_FUNCTIONS
            self.mappings = None
        else:
            raise ValueError(f"Invalid composition type: {self.cfg.composition.type}")
        self.function_names = list(self.function_dict.keys())

    def apply_function_composition_diverse(self, function_list, xstr1, xstr2):
        """
        Applies a function to a string.
        """
        outputs = []
        # apply the function to the string
        for function in function_list:
            if function == "join" or function == "union":
                # if the function is join or intersect, apply it to both strings
                xstr1 = self.function_dict[function](xstr1, xstr2)
            elif function == "map":
                xstr1 = self.function_dict[function](xstr1, self.cfg.n_alphabets)
            else:
                xstr1 = self.function_dict[function](xstr1)

            outputs.append(xstr1)
        return outputs


    def apply_function_composition_uniform(self, function_list, xstr1):
        """
        Applies a function to a string.
        """
        outputs = []
        # apply the function to the string
        for function in function_list:
            xstr1 = self.function_dict[function](xstr1)
            outputs.append(xstr1)
        return outputs

    def apply_function_composition(self, function_list, xstr1, xstr2=None):
        if self.cfg.composition.type == "diverse":
            return self.apply_function_composition_diverse(function_list, xstr1, xstr2)
        elif self.cfg.composition.type == "uniform":
            return self.apply_function_composition_uniform(function_list, xstr1)
        else:
            raise ValueError(f"Invalid function type: {self.cfg.composition.type}")

class CompositionsGenerator(BaseCompositionsClass):
    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg)
        self._init_split_strategy()
        self._init_combination_generator()

    def _init_split_strategy(self):
        """Initialize the split strategy based on configuration."""
        self.split_strategy = get_split_strategy(
            self.cfg, self.function_names
        )

    def _init_combination_generator(self):
        """Initialize the combination generator."""
        self.combination_generator = FunctionCombinationGenerator(self.cfg)
        self.combination_generator.set_function_names(self.function_names)

    def create_compositions(self) -> Tuple[List[List[str]], Dict[Tuple[str, ...], int]]:
        """
        Create function combinations based on the specified strategy.

        Returns:
            Tuple of (all_functions, combination_ids)
        """
        # Generate combinations using the generator
        unique_function_combinations_permutations = self.combination_generator.generate(
            self.cfg.split_strategy
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

    def get_train_test_compositions(
        self,
    ) -> Tuple[List[List[str]], List[List[str]], Dict[Tuple[str, ...], int]]:
        """
        Returns training and test function splits based on the configured strategy.

        Returns:
            Tuple of (train_functions, test_functions, functions_info)
        """
        all_functions, functions_info = self.create_compositions()

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

# test the functions
def main():
    cfg_path = "{}/config/gen/conf.yaml".format(ROOT_DIR)
    # read the config file
    cfg = read_config(cfg_path)
    cfg.prompt_length = "variable"
    cfg.split_strategy = "combination_3"
    cfg.composition.type = "uniform"
    cfg.composition.task_max_length = 7
    # create the functions
    create_functions = CompositionsGenerator(cfg)
    train_functions, test_functions, functions_info = create_functions.get_train_test_compositions()
    print("train_functions", len(train_functions))
    print("test_functions", len(test_functions))
    print("train_functions", train_functions[:5])
    print("test_functions", test_functions[:5])



if __name__ == "__main__":
    main()
