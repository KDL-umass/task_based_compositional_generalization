
import pickle
import plotly.graph_objects as go
import numpy as np
import pandas as pd


def load_representation_results(path):
    with open(
        path, "rb"
    ) as f:
        representation_results = pickle.load(f)
    return representation_results

def get_representation_df(tsne):
    print(tsne.keys())
    embeddings = tsne["embeddings"]
    function_lists = np.array(tsne["function_lists"])
    input_data_strings = np.array(tsne["input_data_tokens"])
    actual_output_strings = np.array(tsne["actual_output_tokens"])
    predicted_output_strings = np.array(tsne["predicted_output_tokens"])
    dataset_labels = np.array(tsne["split_labels"])
    
    df = pd.DataFrame(embeddings, columns=["x", "y"])
    df["dataset"] = dataset_labels
    df["function_list"] = function_lists
    df["input_string"] = input_data_strings
    df["predicted_output_string"] = predicted_output_strings
    df["actual_output_string"] = actual_output_strings

    # filter df to only contain following function lists:
    print(df["function_list"].unique())
    
    return df


def plot_tsne_with_equivalence_class_score(
    df
):
    width = 1200
    height = 800
    point_size = 8
    alpha_val = 0.8

    def mask_for(split):
        return df["dataset"] == split

    fig = go.Figure()
    mask_train = mask_for("train")
    mask_test = mask_for("test")
    
    # Plot each unique function list (task) with a unique color, and differentiate train/test by marker
    from plotly.colors import qualitative

    # Convert function_list to tuple so it can be hashed/used as key
    df["function_list_str"] = df["function_list"].apply(lambda fl: " ".join(map(str, fl)))
    unique_tasks = df["function_list_str"].unique()

    # Use a palette with at least 30 unique colors
    from plotly.colors import qualitative
    # Combine multiple qualitative color scales to get at least 30 unique colors
    color_palette = (
        qualitative.Plotly +
        qualitative.D3 +
        qualitative.Set1
    )
    # Remove duplicates, preserve order, take first 30
    from collections import OrderedDict
    color_palette = list(OrderedDict.fromkeys(color_palette))[:30]
    n_colors = len(color_palette)
    color_palette = qualitative.Plotly
    n_colors = len(color_palette)

    marker_symbols = {
        "train": "square",
        "test": "circle"
    }

    for split in ["train", "test"]:
        split_mask = df["dataset"] == split
        split_df = df[split_mask]
        if not split_df.empty:
            for i, task in enumerate(unique_tasks):
                task_mask = split_df["function_list_str"] == task
                task_df = split_df[task_mask]
                if not task_df.empty:
                    color = color_palette[i % n_colors]
                    symbol = marker_symbols[split]
                    hovertext = [
                        f"{task} (x: {x}, y_pred: {y_pred}, y_actual: {y_actual})"
                        for x, y_pred, y_actual in zip(
                            task_df["input_string"],
                            task_df["predicted_output_string"],
                            task_df["actual_output_string"]
                        )
                    ]
                    fig.add_trace(
                        go.Scattergl(
                            x=task_df["x"],
                            y=task_df["y"],
                            mode="markers",
                            name=f'{"Train" if split=="train" else "Test"}: {task}',
                            marker=dict(
                                symbol=symbol,
                                color=color,
                                size=point_size,
                                opacity=alpha_val,
                                line=dict(width=0.8 if split=='test' else 0.6, color="black"),
                            ),
                            hovertext=hovertext,
                            showlegend=True,
                        )
                    )

    fig.update_layout(
        title=f"Representation of Test Tasks in t-SNE",
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
    
    print(f"tsne_plot_direct.html")
    fig.write_html(f"tsne_plot_direct.html")

if __name__ == "__main__":
    
    rep_path = f"/scratch/workspace/ppruthi_umass_edu-MI/task_based_cg_debugging/task_based_compositional_generalization/models/eval/uniform/fixed/nalph_26_seqlen_6_fnlen_6_taskmaxlen_2/direct/train_combination_2/eval_combination_2/abs/nh6_nl3/seed_0/representation_results_19601.pkl"
    
    representation_results = load_representation_results(rep_path)
    # model_equivalence_class_map = load_model_equivalence_class_map(path_prefix)

    repr_df = get_representation_df(representation_results)

    plot_tsne_with_equivalence_class_score(repr_df)
