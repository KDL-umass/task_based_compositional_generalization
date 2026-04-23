from src.data.composition_generator import CompositionsGenerator
from src.data.corpus_generator.document_generator import DocumentGenerator
from src.evaluation.evaluator import Evaluator
from src.analysis.representation import RepresentationExtractor, perform_tsne_and_visualize
import torch
import numpy as np
import os
import pickle

class RepresentationAnalyzer(Evaluator):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.token_idx = self.token_mgr.token_idx
        self.sep_token = self.token_mgr.sep_token
        self.end_token = self.token_mgr.end_token
        self.token = self.token_mgr.token
        self.compositions_generator = CompositionsGenerator(cfg)
        self.document_generator = DocumentGenerator(
            cfg,
            self.token_mgr,
            self.compositions_generator
        )

    def _to_numpy_doc(self, doc):
        if isinstance(doc, torch.Tensor):
            return doc.detach().cpu().numpy()
        return np.asarray(doc)

    def _get_unique_train_function_lists(self, train_docs):
        unique_fn_lists = set()
        for doc in train_docs:
            doc_np = self._to_numpy_doc(doc)
            fn_tokens = self.token_mgr.get_function_list(doc_np, self.token_map)
            unique_fn_lists.add(tuple(int(tok) for tok in fn_tokens))
        return [list(fn_tokens) for fn_tokens in sorted(unique_fn_lists)]

    def generate_actual_outputs_for_single_sample(self, function_list_tokens, input_str1_tokens, input_str2_tokens):
        function_list = [self.token_mgr.token[tok] for tok in function_list_tokens]
        input_str1 = [self.token_mgr.token[tok] for tok in input_str1_tokens]
        input_str2 = [self.token_mgr.token[tok] for tok in input_str2_tokens] if input_str2_tokens is not None else None
        outputs = self.compositions_generator.apply_function_composition(
            function_list,
            input_str1,
            input_str2
        )

        outputs = self.document_generator._pad_outputs(outputs)
        final_output = outputs[-1]
        outputs_tokens = [self.token_mgr.token_idx[final_output[i]] for i in range(len(final_output))]
        # convert list of list to list 
        
        return outputs_tokens

    def generate_actual_outputs_for_all_samples(self, function_list_tokens, input_str1_tokens, input_str2_tokens):
        outputs_tokens = []
        for function_list_tokens, input_str1_tokens, input_str2_tokens in zip(function_list_tokens, input_str1_tokens, input_str2_tokens):
            outputs_tokens.append(self.generate_actual_outputs_for_single_sample(function_list_tokens, input_str1_tokens, input_str2_tokens))
        return outputs_tokens

    def get_processed_docs(self, docs, unique_train_function_lists):
        doc_inputs = []
        doc_outputs = []
        doc_function_lists = []
        doc_input_data_tokens = []
        
        doc_split_labels = []
        if not hasattr(self, '_seq_info'):
            self._seq_info = self.token_mgr.get_seq_info(
                docs[0]
            )
        
        for doc in docs:
            doc_inputs.append(doc[:self._seq_info["prompt_pos_end"]])
            doc_outputs.append(self.token_mgr.get_output_string(doc, self.cfg.function_type))
            doc_function_lists.append(self.token_mgr.get_function_list(doc))
            doc_input_data_tokens.append(self.token_mgr.get_input_string(doc, self.cfg.function_type))
            doc_split_labels.append("test")
            

            for fn_list in unique_train_function_lists:
                prefix = np.array(
                    [self.token_idx[self.token_mgr.start_token], *fn_list, self.token_idx[self.sep_token]],
                    dtype=doc.dtype,
                )
                
                new_doc = np.concatenate([prefix, doc_input_data_tokens[-1], np.array([self.token_mgr.sep_idx])]).astype(doc.dtype)
                doc_inputs.append(new_doc)
                if self.cfg.function_type == "diverse":
                    sep_pos = np.where(doc_input_data_tokens[-1] == self.sep_token)[0][1]
                    input_str1_tokens, input_str2_tokens = doc_input_data_tokens[-1][:sep_pos], doc_input_data_tokens[-1][sep_pos + 1:]
                else:
                    input_str1_tokens = doc_input_data_tokens[-1]
                    input_str2_tokens = None
                doc_outputs.append(self.generate_actual_outputs_for_single_sample(fn_list, input_str1_tokens, input_str2_tokens))
                doc_function_lists.append(self.token_mgr.get_function_list(new_doc))
                
                doc_input_data_tokens.append(self.token_mgr.get_input_string(new_doc, self.cfg.function_type))
                doc_split_labels.append("train")
        return doc_inputs, doc_outputs, doc_function_lists, doc_input_data_tokens, doc_split_labels

    def extract_all_representations(
        self,
        docs,
        device="cuda",
        batch_size=100,
        split="train",
        unique_train_function_lists=None,
    ):
        """Extract representations for all documents in the dataset"""
        extractor = RepresentationExtractor(self.model, device)
        extractor.register_hook()

        self.logger.info(f"Processing {len(docs)} documents...")
        self.logger.info(f"Found {len(unique_train_function_lists)} unique train function compositions")

        doc_inputs, doc_outputs, doc_function_lists, doc_input_data_tokens, doc_split_labels = self.get_processed_docs(docs, unique_train_function_lists)

        all_representations = []
        all_function_lists = []
        all_input_data_tokens = []
        all_actual_output_tokens = []
        all_predicted_output_tokens = []
        all_split_labels = []
        num_batches = len(doc_inputs) // batch_size + (
                    1 if len(doc_inputs) % batch_size != 0 else 0
                )
        
        # Extract representations
        for batch_idx in range(num_batches):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{num_batches}")

            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(doc_inputs))
            
            batch_inputs = doc_inputs[start_idx:end_idx]
            batch_outputs = doc_outputs[start_idx:end_idx]
            batch_function_lists = doc_function_lists[start_idx:end_idx]
            batch_input_data_tokens = doc_input_data_tokens[start_idx:end_idx]

            last_token_repr_tensor, predicted_output_batch = extractor.extract_representation(
                batch_inputs,
                self._seq_info,
                pad_idx=self.token_mgr.null_idx,
                prompt_mode=self.cfg.prompt_mode,
            )
            predicted_output_batch = predicted_output_batch.detach().cpu().numpy()
            
            all_representations.extend(last_token_repr_tensor)
            all_function_lists.extend(batch_function_lists)
            all_input_data_tokens.extend(batch_input_data_tokens)
            all_predicted_output_tokens.extend(predicted_output_batch)
            all_actual_output_tokens.extend(batch_outputs)
            all_split_labels.extend(doc_split_labels[start_idx:end_idx])
        extractor.remove_hook()

        
        decoded_doc_fns = [self.token_mgr.decode(doc_fn) for doc_fn in all_function_lists]
        decoded_doc_data_tokens = [self.token_mgr.decode(doc_inp_d) for doc_inp_d in all_input_data_tokens]
        decoded_doc_outputs = [self.token_mgr.decode(doc_oup) for doc_oup in all_actual_output_tokens]
        decoded_predicted_doc_outptus = [self.token_mgr.decode(doc_oup) for doc_oup in all_predicted_output_tokens]
        

        return (
            np.array(all_representations),
            decoded_doc_fns,
            decoded_doc_data_tokens,
            decoded_doc_outputs,
            decoded_predicted_doc_outptus,
            all_split_labels,
        )

    def analyze_representations(self):
        """
        Perform representation analysis on train and test data

        Args:
            net: Trained model
            ck: Checkpoint number

        Returns:
            dict: Analysis results including representations and t-SNE embeddings
        """
        results = {}
        ckpts = self.get_latest_ckpt()
        latest_iter, ckpt_file = ckpts[-1]
        self.load_net(ckpt_file)
        self.logger.info(f"Loaded checkpoint: {ckpt_file} for iteration {latest_iter}")

        self.model.eval()
        if "cuda" in self.device:
            self.model = self.model.cuda()

        # Load corpus data
        dataloaders = self.load_dataloaders()
        train_dataset = dataloaders["train"].dataset.data
        test_dataset = dataloaders["test"].dataset.data

        self.logger.info(f"Train data: {len(train_dataset)} samples")
        self.logger.info(f"Test data: {len(test_dataset)} samples")

        # Extract representations for train data
        self.logger.info("Extracting train representations...")
        unique_train_function_lists = self._get_unique_train_function_lists(train_dataset)
        
        # Extract representations for test data
        self.logger.info("Extracting test representations...")
        (
            representations,
            function_lists,
            input_data_tokens,
            actual_output_tokens,
            predicted_output_tokens,
            split_labels,
        ) = self.extract_all_representations(
            test_dataset,
            unique_train_function_lists=unique_train_function_lists,
        )
       
        # Prepare results
        results[latest_iter] = {
            "representations": representations,
            "function_lists": function_lists,
            "input_data_tokens": input_data_tokens,
            "predicted_output_tokens": predicted_output_tokens,
            "actual_output_tokens": actual_output_tokens,
            "split_labels": split_labels,
        }

        # Perform t-SNE analysis
        self.logger.info("Performing t-SNE analysis...")
        embeddings = (
            perform_tsne_and_visualize(
                representations
            )
        )

        # Add t-SNE results
        results[latest_iter].update(
            {
                "embeddings": embeddings
            }
        )

        return results

    def save_representation_results(
        self, results
    ):
        """Save accuracy results"""
        results_path = self.get_output_dir()
        for latest_iter, results in results.items():
            output_path = os.path.join(results_path, f"representation_results_{latest_iter}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            self.logger.info(f"Saved representation results to: {output_path}")