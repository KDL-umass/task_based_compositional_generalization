import argparse
import pickle
from init import read_config, set_seed
from src.data.composition_generator.compositions import CompositionsGenerator
from src.analysis.representation_extractor import CompositionRepresentationExtractor, perform_tsne_and_visualize
from torch.utils.data import DataLoader
# make analyzer inheriting from Evaluator class
from src.evaluation.evaluator import Evaluator
import os
from src.utils.storage_utils import get_directory_path
from init import ROOT_DIR
from src.utils.logging_utils import setup_analysis_logging
class Analyzer(Evaluator):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.logger = logger
        self.extractor = CompositionRepresentationExtractor(self.cfg, self.logger)

    def analyze_representations(self, model, docs):
        extractor = self.extractor
        representations, function_lists, input_strings, output_strings = extractor.extract_all_representations(model, docs)
        return representations, function_lists, input_strings, output_strings

    def save_representation_results(self, results):
        """Save accuracy results"""
        output_dir = self.get_output_dir()
        os.makedirs(output_dir, exist_ok=True)
        pickle.dump(results, open(output_dir + f"/representation_results.pkl", "wb"))

    def analyze(self):
        ck, ckpt_file = self.get_latest_ckpt()
        self.load_net(ckpt_file)
        dataloaders = self.load_dataloaders()
        train_dataset = dataloaders["train"].dataset.data
        train_dataloader = self.create_uniform_sampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.data.batch_size,
            sampler=train_dataloader,
            num_workers=self.cfg.data.num_workers
        )
        results = {}
        for split in ["train", "train_heldout", "test"]:
            dataset = dataloaders[split].dataset.data
            representations, function_lists, input_strings, output_strings = self.analyze_representations(self.model, dataset)
            results[split] = {
                "representations": representations,
                "function_lists": function_lists,
                "input_strings": input_strings,
                "output_strings": output_strings
            }
        embeddings, perm_labels, dataset_labels, unique_perms, simplified_perms = perform_tsne_and_visualize(results)
        results["embeddings"] = embeddings
        results["perm_labels"] = perm_labels
        results["dataset_labels"] = dataset_labels
        results["unique_perms"] = unique_perms
        results["simplified_perms"] = simplified_perms
        self.save_representation_results(results)
    



def main(cfg):
    print("Running analyzer with the following configuration:")
    # print config in a readable format
    print(cfg)
    set_seed(cfg.seed)
    logger = setup_analysis_logging(cfg)
    analyzer = Analyzer(cfg, logger)
    analyzer.analyze()
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_mode", type=str, default="step_by_step", help="step or direct"
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="coverage_6_0_0",
        help="Training split",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="coverage_6_0_0",
        help="Model evaluation split",
    )
    parser.add_argument(
        "--nheads_nlayers",
        type=str,
        default="nh6_nl3",
        help="number of heads and layers",
    )
    parser.add_argument(
        "--pos_embedding_type", type=str, default="rel_global", help="abs or rel_global"
    )
    parser.add_argument("--num_runs", type=int, default=1, help="number of runs")
    parser.add_argument(
        "--function_type", type=str, default="uniform", help="uniform or diverse"
    )
    parser.add_argument(
        "--task_max_length", type=int, default=6, help="max task length"
    )
    
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained", type=bool, default=False)
    args = parser.parse_args()

    # over-ride some of the arguments based on the run-time args
    if args.pretrained:
        cfg = read_config(f"{ROOT_DIR}/config/eval/conf_finetune.yaml")
    else:
        cfg = read_config(f"{ROOT_DIR}/config/eval/conf.yaml") 
    cfg.pretrained = args.pretrained
    cfg.prompt_mode = args.prompt_mode
    cfg.train_split = args.train_split
    cfg.eval_split = args.eval_split
    cfg.nheads_nlayers = args.nheads_nlayers
    cfg.pos_embedding_type = args.pos_embedding_type
    cfg.num_runs = args.num_runs
    cfg.function_type = args.function_type
    cfg.seed = args.seed
    cfg.task_max_length = args.task_max_length
    cfg.data_path = get_directory_path(cfg, key='data', prefix_dir='data')
    cfg.data_path = os.path.join(cfg.data_path, cfg.prompt_mode, cfg.train_split)
    main(cfg)
