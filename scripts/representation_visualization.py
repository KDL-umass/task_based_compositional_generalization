import argparse
import pickle
from init import read_config, set_seed
# make analyzer inheriting from Evaluator class
from src.evaluation.evaluator import Evaluator
import os
from src.utils.storage_utils import get_directory_path
from init import ROOT_DIR
from src.utils.logging_utils import setup_visualization_logging
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
class RepresentationVisualizer(Evaluator):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.logger = logger


    def load_representation_results(self):
        output_dir = self.get_output_dir()
        representation_results = pickle.load(open(output_dir + f"/representation_results.pkl", "rb"))
        return representation_results

    def get_representation_df(self):
        representation_results = self.load_representation_results()
        embeddings = representation_results["embeddings"]
        perm_labels = representation_results["perm_labels"]
        dataset_labels = representation_results["dataset_labels"]
        unique_perms = representation_results["unique_perms"]
        simplified_perms = representation_results["simplified_perms"]
        all_input_strings = representation_results["train"]["input_strings"] + representation_results["train_heldout"]["input_strings"] + representation_results["test"]["input_strings"]
        all_output_strings = representation_results["train"]["output_strings"] + representation_results["train_heldout"]["output_strings"] + representation_results["test"]["output_strings"]
        all_function_lists = representation_results["train"]["function_lists"] + representation_results["train_heldout"]["function_lists"] + representation_results["test"]["function_lists"]
        print(len(all_input_strings), len(all_output_strings), len(all_function_lists))
        print(len(perm_labels), len(simplified_perms), len(unique_perms))
        print(len(representation_results["train"]["input_strings"]), len(representation_results["train_heldout"]["input_strings"]), len(representation_results["test"]["input_strings"]))
     
        df = pd.DataFrame(embeddings, columns=["x", "y"])
        df["perm_label"] = perm_labels
        df["perm_tuple"] = [tuple(row) for row in simplified_perms]
        df["dataset"] = dataset_labels
        # df["input_string"] = all_input_strings
        df["output_string"] = all_output_strings
        df["function_list"] = all_function_lists
        print(df.head())
        
        return df
    
    def visualize_representations(self):
        df = self.get_representation_df()
        self.plot_tsne(df)

    def plot_tsne(self,
        df
    ):
        width = 1200
        height = 800
        point_size = 8
        alpha_val = 0.8

        def mask_for(highlight_set):
            return df["perm_tuple"].isin(highlight_set)

        fig = go.Figure()
        
        train_keys = df[df["dataset"] == "train"]["perm_tuple"].unique()
        test_keys = df[df["dataset"] == "test"]["perm_tuple"].unique()
        train_mask = mask_for(train_keys)
        test_mask = mask_for(test_keys)
        

        # Plot Level 3: light gray diamonds (background)
        if train_mask.any():
            mask1_subset = df[train_mask]

            fig.add_trace(
                go.Scattergl(
                    x=mask1_subset["x"],
                    y=mask1_subset["y"],
                    mode="markers",
                    name="Train Tasks",
                    marker=dict(
                        symbol="diamond", color="lightgrey", size=point_size, opacity=0.25
                    ),
                    hovertext=[
                        f"perm: {' '.join(list(perm))}<br>output: {out}"
                        for perm, out in zip(
                            mask1_subset["perm_tuple"],
                            mask1_subset["output_string"],
                        )
                    ],
                    showlegend=True,
                )
            )
        if test_mask.any():
            mask2_subset = df[test_mask]
            fig.add_trace(
                go.Scattergl(
                    x=mask2_subset["x"],
                    y=mask2_subset["y"],
                    mode="markers",
                    name="Test Tasks",
                    marker=dict(
                        symbol="circle", color="red", size=point_size, opacity=0.25
                    ),
                    hovertext=[
                        f"perm: {' '.join(list(perm))}<br>output: {out}"
                        for perm, out in zip(
                            mask2_subset["perm_tuple"],
                            mask2_subset["output_string"],
                        )
                    ],
                    showlegend=True,
                )
            )

        

        fig.update_layout(
            title=f"Representation of Train and Test Tasks in t-SNE",
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            width=width,
            height=height,
            legend=dict(
                title="Legend",
                x=1.02,
                y=1,
                xanchor="left",
                yanchor="top",
                font=dict(size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            margin=dict(r=300, l=50, t=50, b=50),  # Increased right margin for color bars
        )
        
        print(f"tsne_plot.html")
        fig.write_html(f"{self.get_output_dir()}/tsne_plot.html")


def main(cfg):
    print("Running representation visualizer with the following configuration:")
    # print config in a readable format
    print(cfg)
    set_seed(cfg.seed)
    logger = setup_visualization_logging(cfg)
    visualizer = RepresentationVisualizer(cfg, logger)
    visualizer.visualize_representations()

    

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


    