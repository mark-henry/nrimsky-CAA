"""
Script to plot PCA of constrastive activations

Usage:
python plot_activations.py --behaviors sycophancy --layers 9 10 11 --model meta-llama/Llama-2-7b-chat-hf --use_chat
python plot_activations.py --behaviors sycophancy --layers 9 10 11 --model google/gemma-2-9b --use_chat
"""

import json
import torch as t
import os
from matplotlib import pyplot as plt
import argparse
from sklearn.decomposition import PCA
from behaviors import HUMAN_NAMES, get_activations_path, get_ab_data_path, get_analysis_dir, ALL_BEHAVIORS
from utils.helpers import set_plotting_settings
from tqdm import tqdm

DATASET_FILE = os.path.join("preprocessed_data", "generate_dataset.json")

set_plotting_settings()

def save_activation_projection_pca(behavior: str, layer: int, model_name_path: str, use_chat: bool):
    model_short_name = model_name_path.split("/")[-1]
    title = f"{HUMAN_NAMES[behavior]}, {model_short_name}, layer {layer}"
    fname = f"pca_{behavior}_layer_{layer}_{model_short_name}_chat_{use_chat}.png"
    save_dir = os.path.join(get_analysis_dir(behavior), "pca", model_short_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loading activations
    activations_pos = t.load(
        get_activations_path(behavior, layer, model_name_path, "pos")
    )
    activations_neg = t.load(
        get_activations_path(behavior, layer, model_name_path, "neg")
    )

    # Getting letters
    with open(get_ab_data_path(behavior), "r") as f:
        data = json.load(f)

    letters_pos = [item["answer_matching_behavior"][1] for item in data]
    letters_neg = [item["answer_not_matching_behavior"][1] for item in data]

    plt.clf()
    plt.figure(figsize=(6, 6))
    activations = t.cat([activations_pos, activations_neg], dim=0)
    activations_np = activations.cpu().numpy()

    # PCA projection
    pca = PCA(n_components=2)
    projected_activations = pca.fit_transform(activations_np)

    # Splitting back into activations1 and activations2
    activations_pos_projected = projected_activations[: activations_pos.shape[0]]
    activations_neg_projected = projected_activations[activations_pos.shape[0] :]

    # Visualization
    for i, (x, y) in enumerate(activations_pos_projected):
        if letters_pos[i] == "A":
            plt.scatter(x, y, color="blue", marker="o", alpha=0.4)
        elif letters_pos[i] == "B":
            plt.scatter(x, y, color="blue", marker="x", alpha=0.4)

    for i, (x, y) in enumerate(activations_neg_projected):
        if letters_neg[i] == "A":
            plt.scatter(x, y, color="red", marker="o", alpha=0.4)
        elif letters_neg[i] == "B":
            plt.scatter(x, y, color="red", marker="x", alpha=0.4)

    # Adding the legend
    scatter1 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        label=f"pos {HUMAN_NAMES[behavior]} - A",
    )
    scatter2 = plt.Line2D(
        [0],
        [0],
        marker="x",
        color="blue",
        markerfacecolor="blue",
        markersize=10,
        label=f"pos {HUMAN_NAMES[behavior]} - B",
    )
    scatter3 = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=10,
        label=f"neg {HUMAN_NAMES[behavior]} - A",
    )
    scatter4 = plt.Line2D(
        [0],
        [0],
        marker="x",
        color="red",
        markerfacecolor="red",
        markersize=10,
        label=f"neg {HUMAN_NAMES[behavior]} - B",
    )

    plt.legend(handles=[scatter1, scatter2, scatter3, scatter4])
    plt.title(title)
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), format="png", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--behaviors",
        nargs="+",
        type=str,
        default=ALL_BEHAVIORS,
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        required=True,
    )
    parser.add_argument("--use_chat", action="store_true",
                        help="whether to use chat-style prompting", default=False)
    parser.add_argument("--model", type=str, required=True,
                        help="e.g. google/gemma-2b-it, meta-llama/Llama-2-7b-chat-hf")
    args = parser.parse_args()

    for behavior in args.behaviors:
        print(f"Plotting {behavior} activations PCA for {args.model}")
        for layer in tqdm(args.layers):
            save_activation_projection_pca(
                behavior,
                layer,
                args.model,
                args.use_chat
            )