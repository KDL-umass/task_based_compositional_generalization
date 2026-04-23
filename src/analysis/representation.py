"""
Representation extraction utilities for model analysis
"""

import os
import pickle
from collections import defaultdict
import torch

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


class RepresentationExtractor:
    """Extract representations from transformer models for analysis"""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.last_layer_repr = None
        self.hook = None

    def hook_fn(self, module, input, output):
        """Hook function to capture last layer representations"""
        self.last_layer_repr = output.detach().cpu()

    def register_hook(self):
        """Register hook on the final layer norm"""
        self.hook = self.model.transformer.ln_f.register_forward_hook(self.hook_fn)

    def remove_hook(self):
        """Remove the hook"""
        if self.hook:
            self.hook.remove()

    def extract_representation(
        self,
        input_batch,
        seq_info,
        *,
        pad_idx: int = 0,
        sep_idx: int | None = None,
        function_type: str = "uniform",
        prompt_mode: str = "direct",
        fixed_output_len: int | None = None,
    ):
        """Extract representation for a single batch"""
        self.model.eval()
        all_representations = []
        # Prompts can have variable lengths (esp. `step_by_step`). We left-pad so the
        # final real prompt token is aligned at the same position for the whole batch.
        lengths = [len(x) for x in input_batch]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            raise ValueError("Empty input_batch passed to extract_representation()")

        padded = np.full((len(input_batch), max_len), pad_idx, dtype=np.int64)
        left_pad_offsets = []
        for i, x in enumerate(input_batch):
            arr = np.asarray(x, dtype=np.int64)
            offset = max_len - arr.shape[0]
            left_pad_offsets.append(offset)
            padded[i, offset:] = arr

        input_batch = torch.as_tensor(padded, device=self.device)
        with torch.no_grad():
            # Extract input data token representations (one forward pass on the prompt)
            _ = self.model(input_batch)
            # Determine the slice of tokens corresponding to the input strings.
            # In `step_by_step`, there are multiple <SEP>s, so we *cannot* use the
            # last separator; we want just the raw input(s) portion.
            if sep_idx is None:
                input_data_start = seq_info["input_data_start"]
                input_data_end = seq_info["input_data_end"]
                input_repr = self.last_layer_repr[:, input_data_start:input_data_end, :]
            else:
                # Per-sample slicing, then stack (input lengths are fixed by data config).
                per_sample_input_repr = []
                for i, raw in enumerate(input_batch.detach().cpu().numpy()):
                    # Recover the unpadded sequence by stripping left pad.
                    offset = left_pad_offsets[i]
                    unpadded = raw[offset:]
                    sep_positions = np.where(unpadded == sep_idx)[0]
                    if len(sep_positions) < 2:
                        raise ValueError(
                            f"Expected at least 2 <SEP> tokens, found {len(sep_positions)}"
                        )
                    start = int(sep_positions[0]) + 1
                    if function_type in ["diverse", "diverse2"]:
                        if len(sep_positions) < 3:
                            raise ValueError(
                                f"Expected at least 3 <SEP> tokens for {function_type}, "
                                f"found {len(sep_positions)}"
                            )
                        end = int(sep_positions[2])
                    else:
                        end = int(sep_positions[1])

                    start_t = start + offset
                    end_t = end + offset
                    per_sample_input_repr.append(
                        self.last_layer_repr[i : i + 1, start_t:end_t, :]
                    )
                input_repr = torch.cat(per_sample_input_repr, dim=0)

            input_repr_flat = input_repr.reshape(input_repr.shape[0], -1)
            # Shape: (batch_size, input_data_len * hidden_dim)

            # Generation length:
            # - direct: use seq_info (prompt ends at last SEP before output)
            # - step_by_step: we typically want the *final* output only, so use a fixed length
            if prompt_mode == "step_by_step" and fixed_output_len is not None:
                gen_len = int(fixed_output_len)
            else:
                gen_len = int(seq_info["new_len"])

            for _ in range(gen_len):
                logits = self.model(input_batch)
                logits = logits[:, -1, :]
                inp_next = torch.argmax(logits, -1, keepdims=True)
                input_batch = torch.cat((input_batch, inp_next), dim=1)
                final_representation = self.last_layer_repr[:, -1, :]
                final_representation = final_representation.unsqueeze(1)
                all_representations.append(final_representation)

        all_representations = torch.cat(all_representations, dim=1)
        all_representations = all_representations.reshape(
            all_representations.shape[0], -1
        )
        # Concatenate input data repr with output repr
        combined_representations = torch.cat([input_repr_flat, all_representations], dim=1)

        # With left-padding, generated tokens always begin at `max_len`.
        return combined_representations, input_batch[:, max_len:]


def create_unique_function_permutations(all_function_lists):
    """Create unique permutations from function lists and assign colors"""
    # Remove identity functions and create tuples
    simplified_permutations = []
    for fn_list in all_function_lists:
        simplified = tuple(fn for fn in fn_list if fn != "identity")
        simplified_permutations.append(simplified)

    # Get unique permutations
    unique_permutations = list(set(simplified_permutations))

    # Create color mapping - convert tuples to strings for LabelEncoder
    simplified_perms_strings = [str(perm) for perm in simplified_permutations]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(simplified_perms_strings)

    return simplified_permutations, unique_permutations, labels, label_encoder


def perform_tsne_and_visualize(
    representations,
    n_components=2,
    perplexity=30,
    random_state=42,
):
    """Perform t-SNE and create visualization"""

    # Combine all representations
    all_representations = representations
    # Perform t-SNE
    print("Performing t-SNE...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=1,
    )
    embeddings = tsne.fit_transform(all_representations)

    return embeddings


def save_representation_results(results, save_path):
    """Save representation analysis results"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {save_path}")


def load_representation_results(load_path):
    """Load representation analysis results"""
    with open(load_path, "rb") as f:
        results = pickle.load(f)
    print(f"Results loaded from {load_path}")
    return results
