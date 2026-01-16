"""
Representation extraction utilities for model analysis
"""

import torch
import numpy as np
from src.data.corpus_generator.token_manager import TokenManager
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


class RepresentationExtractor:
    """Extract representations from transformer models for analysis"""
    
    def __init__(self, model, device='cuda'):
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
    
    def extract_representation(self, input_batch, seq_info):
        """Extract representation for a single batch"""
        self.model.eval()
        self.model.to(self.device)
        input_batch = input_batch.to(self.device)
        all_representations = []
        with torch.no_grad():
            for _ in range(seq_info["new_len"]):    
                logits = self.model(input_batch)
                logits = logits[:, -1, :]
                inp_next = torch.argmax(logits, -1, keepdims=True)
                input_batch = torch.cat((input_batch, inp_next), dim=1)
                final_representation = self.last_layer_repr[:,-1,:]
                # create a new dimension in between 1 and 0
                final_representation = final_representation.unsqueeze(1)
                all_representations.append(final_representation)
        all_representations = torch.cat(all_representations, dim=1)
        all_representations = all_representations.reshape(all_representations.shape[0], -1)
        
        return all_representations, input_batch[:, seq_info["prompt_pos_end"]:]


class CompositionRepresentationExtractor:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.token_mgr = TokenManager(load_path=cfg.data_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logger

    def _extract_inputs_from_docs(self, docs):
        """Extract input strings from a test document that uses the target function list"""
        
        # Find a document with this function list
        string_inputs = []
        for doc in docs:
            string_inputs.append(self._extract_inputs_from_doc(doc))
        return string_inputs

    def _extract_inputs_from_doc(self, doc, tokens_to_string = False):
        """Extract input strings from document"""
        sep_positions = np.where(doc == self.token_mgr.sep_idx)[0]
        
        if len(sep_positions) < 2:
            raise ValueError("Document doesn't have enough SEP tokens")
        
        # First input: between first and second SEP
        input1_tokens = doc[sep_positions[0] + 1:sep_positions[1]]
        if tokens_to_string:
            input_str1 = self._tokens_to_string(input1_tokens)
        else:
            input_str1 = input1_tokens
        
        # Second input: between second and third SEP (if exists)
        input_str2 = ""
        if len(sep_positions) >= 3:
            input2_tokens = doc[sep_positions[1] + 1:sep_positions[2]]
            if tokens_to_string:
                input_str2 = self._tokens_to_string(input2_tokens)
            else:
                input_str2 = input2_tokens
        
        return input_str1, input_str2

    def _tokens_to_string(self, tokens):
        """Convert token indices to string"""
        # convert tokens to numpy array
        tokens = np.array(tokens)
        result = ""
        for token in tokens:
            char = self.token_mgr.token[token]
            if len(char) == 1 and char.isalpha():  # Only alphabetic characters
                result += char
        return result

    def extract_all_representations(self, net, docs, device='cuda', batch_size=100):
        """Extract representations for all documents in the dataset"""
        extractor = RepresentationExtractor(net, device)
        extractor.register_hook()
        
        all_representations = []
        all_function_lists = []
        input_strings = []
        output_strings = []
        
        self.logger.info(f"Processing {len(docs)} documents for {self.cfg.prompt_mode} split...")
        
        num_batches = len(docs) // batch_size + (1 if len(docs) % batch_size != 0 else 0)
        
        for batch_idx in range(num_batches):
            if batch_idx % 10 == 0:
                self.logger.info(f"Processing batch {batch_idx}/{num_batches} for {self.cfg.prompt_mode} split...")
            
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(docs))
            
            batch_docs = docs[start_idx:end_idx]
            batch_seq_infos = [self.token_mgr.get_seq_info(doc, self.cfg.function_type) for doc in batch_docs]
            
            # Prepare input batch with random strings or use existing strings
            batch_inputs = []
            input_strings = []
            
            
            for doc, seq_info in zip(batch_docs, batch_seq_infos):
                new_doc = doc[:seq_info["prompt_pos_end"]]
                batch_inputs.append(new_doc)
                input_str1, input_str2 = self._extract_inputs_from_doc(doc, tokens_to_string=True)
                
                input_strings.append(input_str1 + input_str2)
            max_len = max(len(inp) for inp in batch_inputs)
            padded_inputs = [
                np.pad(inp, (0, max_len - len(inp)), 'constant') 
                for inp in batch_inputs
            ]
            padded_inputs = np.array(padded_inputs)
            inp_batch = torch.tensor(padded_inputs).to(self.device)
            
            # Extract representations
            last_token_repr_tensor, output_batch = extractor.extract_representation(inp_batch, batch_seq_infos[0])
            output_batch = output_batch.detach().cpu().numpy()
            
            # decode the output batch
            decoded_output_batch = []
            for i in range(len(output_batch)):
                decoded_output_batch.append([self.token_mgr.token[j] for j in output_batch[i]])
            output_strings.extend(decoded_output_batch)
            
            # Take the last token representations
            all_representations.extend(last_token_repr_tensor)
            
            # Extract function lists
            batch_fn_lists = [self.token_mgr.get_function_list(doc) for doc in batch_docs]
            batch_function_lists = [
                [self.token_mgr.token[fn] for fn in fn_list] 
                for fn_list in batch_fn_lists
            ]
            all_function_lists.extend(batch_function_lists)
        
        extractor.remove_hook()
        
        return np.array(all_representations), all_function_lists, input_strings, output_strings

def create_unique_function_permutations(all_function_lists):
    """Create unique permutations from function lists and assign colors"""
    # Remove identity functions and create tuples
    simplified_permutations = []
    for fn_list in all_function_lists:
        simplified = tuple(fn for fn in fn_list if fn != 'identity')
        simplified_permutations.append(simplified)
    
    # Get unique permutations
    unique_permutations = list(set(simplified_permutations))
    
    # Create color mapping - convert tuples to strings for LabelEncoder
    simplified_perms_strings = [str(perm) for perm in simplified_permutations]
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(simplified_perms_strings)
    
    return simplified_permutations, unique_permutations, labels, label_encoder

def perform_tsne_and_visualize(results,
                            save_path=None, n_components=2, perplexity=30, random_state=42):
    """Perform t-SNE and create visualization"""
    
    # Combine all representations
    all_representations = np.vstack([results["train"]["representations"], results["train_heldout"]["representations"], results["test"]["representations"]])
    all_function_lists = results["train"]["function_lists"] + results["train_heldout"]["function_lists"] + results["test"]["function_lists"]
    
    # Create dataset labels (train vs test)
    dataset_labels = ['train'] * len(results["train"]["representations"]) + ['train_heldout'] * len(results["train_heldout"]["representations"]) + ['test'] * len(results["test"]["representations"])
    
    # Create unique function permutations and labels
    simplified_perms, unique_perms, perm_labels, label_encoder = create_unique_function_permutations(all_function_lists)
    
    print(f"Found {len(unique_perms)} unique function permutations")
    print(f"Total representations: {len(all_representations)}")
    
    # Perform t-SNE
    print("Performing t-SNE...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, 
                n_iter=1000, verbose=1)
    embeddings = tsne.fit_transform(all_representations)
    
    return embeddings, perm_labels, dataset_labels, unique_perms, simplified_perms