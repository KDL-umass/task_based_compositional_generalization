"""Detailed error analysis for model evaluation."""
import numpy as np
import torch
from collections import defaultdict

from src.evaluation.utils import is_ood_prompt
from src.data_generation.utils import get_function_list as get_function_list_util


class ErrorAnalyzer:
    """Handles detailed error analysis including step-by-step and module-wise metrics."""
    
    def __init__(self, dictionary, token, token_idx, sep_token, eval_cfg, decode_fn):
        """
        Initialize error analyzer.
        
        Args:
            dictionary: TokenManager instance
            token: Token dictionary mapping indices to tokens
            token_idx: Reverse token dictionary mapping tokens to indices
            sep_token: Separator token string
            eval_cfg: Evaluation configuration
            decode_fn: Function to decode token indices to strings
        """
        self.dictionary = dictionary
        self.token = token
        self.token_idx = token_idx
        self.sep_token = sep_token
        self.eval_cfg = eval_cfg
        self.decode = decode_fn
        
    def analyze_step_by_step(self, dat, output_l, targets_l):
        """Optimized step-by-step error analysis using vectorized operations"""
        output_l = output_l.cpu().numpy()
        targets_l = targets_l.cpu().numpy()
        dat_np = dat.cpu().numpy()

        batch_size = dat.shape[0]
        pad_length = 2 * self.eval_cfg.seq_len

        # Pre-allocate structures
        module_wise_acc = defaultdict(lambda: {"acc": [], "total": 0})
        step_by_step_acc = {
            "individual": defaultdict(list),
            "cumulative": defaultdict(list),
        }

        # Process all samples in batch
        all_function_lists = []
        all_decoded_lists = []
        max_functions = 0

        # First pass: extract all function lists and find max length
        for i in range(batch_size):
            function_list = get_function_list_util(dat_np[i], self.token_idx[self.sep_token])
            decoded_function_list = self.decode(function_list, return_list=True)
            all_function_lists.append(function_list)
            all_decoded_lists.append(decoded_function_list)
            max_functions = max(max_functions, len(decoded_function_list))

        # Create batch accuracy matrix: [batch_size, max_functions]
        batch_accuracies = np.full((batch_size, max_functions), False, dtype=bool)

        # Vectorized slice processing
        for j in range(max_functions):
            # Determine slice positions for this function step
            if j == 0:
                start_pos = 0
                end_pos = pad_length
            else:
                start_pos = j * (pad_length + 1)  # +1 for SEP token
                end_pos = start_pos + pad_length

            # Extract batch slices for this function step
            valid_samples = []
            for i in range(batch_size):
                if j < len(all_decoded_lists[i]):
                    valid_samples.append(i)

            if not valid_samples:
                continue

            # Vectorized accuracy calculation for valid samples
            valid_samples = np.array(valid_samples)
            batch_slice_output = output_l[valid_samples, start_pos:end_pos]
            batch_slice_targets = targets_l[valid_samples, start_pos:end_pos]

            # Check if all tokens in each sequence match
            slice_accuracies = (batch_slice_output == batch_slice_targets).all(axis=1)

            # Update batch accuracy matrix
            batch_accuracies[valid_samples, j] = slice_accuracies

            # Update step-by-step individual accuracies
            step_by_step_acc["individual"][j].extend(slice_accuracies.tolist())

            # Update module-wise accuracies
            for idx, sample_idx in enumerate(valid_samples):
                fn = all_decoded_lists[sample_idx][j]
                module_wise_acc[fn]["acc"].append(slice_accuracies[idx])
                module_wise_acc[fn]["total"] += 1

        # Vectorized cumulative accuracy calculation
        for j in range(max_functions):
            if j == 0:
                # First step: same as individual
                step_by_step_acc["cumulative"][j] = step_by_step_acc["individual"][
                    j
                ].copy()
            else:
                # Cumulative: AND with all previous steps
                cumulative_acc = []
                for i in range(batch_size):
                    if j < len(all_decoded_lists[i]):
                        # Take AND of all steps up to j
                        cum_acc = batch_accuracies[i, : j + 1].all()
                        cumulative_acc.append(cum_acc)
                step_by_step_acc["cumulative"][j] = cumulative_acc

        return dict(module_wise_acc), step_by_step_acc

    def calculate_step_by_step_metrics(self, dat, output, seq_info, combination_ids):
        """Calculate step-by-step error metrics"""
        # Vectorized slicing
        output_l = output[:, seq_info["prompt_pos_end"] :]
        targets_l = dat[:, seq_info["prompt_pos_end"] :]

        # Vectorized accuracy calculation
        acc_l = output_l == targets_l
        sharp_acc = acc_l.all(dim=-1).float().mean().cpu().numpy()

        # Batch OOD detection
        ood_flags = is_ood_prompt(
            self.token, self.token_idx, output_l, targets_l, self.eval_cfg.prompt_length
        )
        ood_flags = ood_flags.cpu().tolist()
        ood_mean = np.array(ood_flags).mean()

        # Vectorized combination accuracy
        from src.evaluation.utils import calculate_combination_accuracy
        (
            sharp_combination_acc,
            ood_combination_acc,
            total_unique_combination_ids,
            print_error_indices,
        ) = calculate_combination_accuracy(
            acc_l, ood_flags, combination_ids, use_sharp=True
        )

        # Optimized module-wise and step-by-step analysis
        module_wise_acc, step_by_step_acc = self.analyze_step_by_step(
            dat, output_l, targets_l
        )

        # Calculate direct metrics once and reuse
        direct_acc_metrics = self.calculate_direct_metrics(
            dat, output, seq_info, combination_ids
        )

        return {
            "total": {"acc": sharp_acc, "ood": ood_mean},
            "combination": {"acc": sharp_combination_acc, "ood": ood_combination_acc},
            "module_wise": module_wise_acc,
            "step_by_step": step_by_step_acc,
            "direct": direct_acc_metrics,
            "total_unique_combination_ids": total_unique_combination_ids,
            "print_error_indices": print_error_indices,
        }

    def calculate_direct_metrics(self, dat, output, seq_info, combination_ids):
        """Calculate direct error metrics"""
        # Vectorized slicing
        start_idx = seq_info["last_sep_pos"] + 1
        end_idx = seq_info["end_pos"]
        output_l = output[:, start_idx:end_idx]
        targets_l = dat[:, start_idx:end_idx]

        # Vectorized accuracy calculation
        acc_l = output_l == targets_l
        sharp_acc = acc_l.all(dim=-1).float().mean().cpu().numpy()

        # Batch OOD detection
        ood_flags = is_ood_prompt(
            self.token, self.token_idx, output_l, targets_l, self.eval_cfg.prompt_length
        )
        ood_flags = ood_flags.cpu().tolist()
        ood_mean = np.array(ood_flags).mean()

        # Vectorized combination accuracy
        from src.evaluation.utils import calculate_combination_accuracy
        (
            sharp_combination_acc,
            ood_combination_acc,
            total_unique_combination_ids,
            print_error_indices,
        ) = calculate_combination_accuracy(
            acc_l, ood_flags, combination_ids, use_sharp=True
        )

        return {
            "total": {"acc": sharp_acc, "ood": ood_mean},
            "combination": {"acc": sharp_combination_acc, "ood": ood_combination_acc},
            "total_unique_combination_ids": total_unique_combination_ids,
            "print_error_indices": print_error_indices,
        }

