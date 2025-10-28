import numpy as np
from src.data_generation.functions import CreateFunctions, apply_function_composition

class CEMetricCalculator:
    def __init__(self, cfg):
        self.cfg = cfg
        if self.cfg.function.type == "diverse":
            self.max_seq_len = 2 * self.cfg.seq_len
        else:
            self.max_seq_len = self.cfg.seq_len

    def sample_string(self):
        """
        Samples a string of the given length.
        """
        n_alphabets = self.cfg.n_alphabets
        seq_len = self.cfg.seq_len
        with_replacement = self.cfg.with_replacement
        alph = [chr(i + 97) for i in range(n_alphabets)]
        tokens = np.random.choice(alph, size=seq_len, replace=with_replacement)
        return "".join(tokens)

    def calculate_ce_metric(self, train_functions, test_functions, function_dict, N_samples=1000):
        ce_metric = 0
        # Complexity of the algorithm is O(N * M * L), 
        # where N is the number of train functions, 
        # M is the number of test functions, 
        # and N_samples is the number of samples to calculate the CE metric * number of functions in each composition.
        ce_metric_dict = {}
        for train_function in train_functions:
            for test_function in test_functions:
                pairwise_ce = self._calculate_ce_metric_for_single_function(train_function, test_function, self.cfg.function.type, function_dict, N_samples=N_samples)
                pair = (tuple(train_function), tuple(test_function))
                ce_metric_dict[pair] = pairwise_ce
                ce_metric += pairwise_ce
        ce_metric = ce_metric/(len(train_functions) * len(test_functions))
        return  ce_metric, ce_metric_dict

    def _calculate_ce_metric_for_single_function(self, train_function, test_function, function_type, function_dict, N_samples=10000):
        ce_metric = 0
        for i in range(N_samples):
            input_string_1, input_string_2 = self.sample_string(), self.sample_string()
            train_output = apply_function_composition(function_type, train_function, function_dict, input_string_1, input_string_2)
            test_output = apply_function_composition(function_type, test_function, function_dict, input_string_1, input_string_2)
            ce_metric += train_output[-1][:self.max_seq_len] == test_output[-1][:self.max_seq_len]
        return ce_metric / N_samples
