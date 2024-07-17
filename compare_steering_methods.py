import os
import json
import argparse
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from steering_settings import SteeringSettings
from behaviors import get_results_dir, HUMAN_NAMES, ALL_BEHAVIORS


def get_data(layer: str, multiplier: float, settings: SteeringSettings) -> List[Dict[str, Any]]:
    directory = get_results_dir(settings.behavior)
    suffix = settings.make_result_save_suffix(layer=layer, multiplier=multiplier)
    filename = os.path.join(directory, f"results_{suffix}.json")

    if not os.path.exists(filename):
        print(f"Warning: File not found - {filename}")
        return []

    with open(filename, 'r') as f:
        return json.load(f)


def get_avg_key_prob(results: List[Dict[str, Any]], key: str) -> float:
    match_key_prob_sum = 0.0
    for result in results:
        matching_value = result[key]
        denom = result["a_prob"] + result["b_prob"]
        if "A" in matching_value:
            match_key_prob_sum += result["a_prob"] / denom
        elif "B" in matching_value:
            match_key_prob_sum += result["b_prob"] / denom
    return match_key_prob_sum / len(results)


def plot_behavior_comparison(settings: SteeringSettings, multipliers: List[float], layer: int):
    plt.figure(figsize=(10, 6))

    # Plot single layer results
    single_layer_results = []
    for multiplier in multipliers:
        results = get_data(str(layer), multiplier, settings)
        avg_prob = get_avg_key_prob(results, "answer_matching_behavior")
        single_layer_results.append(avg_prob)

    plt.plot(multipliers, single_layer_results, label=f"Layer {layer} Steering", marker='o')

    # Plot all-layers results
    all_layers_results = []
    for multiplier in multipliers:
        results = get_data("multi", multiplier, settings)
        avg_prob = get_avg_key_prob(results, "answer_matching_behavior")
        all_layers_results.append(avg_prob)

    plt.plot(multipliers, all_layers_results, label="All Layers Steering", marker='s')

    plt.xlabel("Steering Vector Multiplier")
    plt.ylabel(f"Probability of {HUMAN_NAMES[settings.behavior]} Response")
    plt.title(f"{HUMAN_NAMES[settings.behavior]} Steering: Layer {layer} vs All Layers\n({settings.model_name_path})")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(get_results_dir(settings.behavior),
                             f"{settings.behavior}_comparison_{settings.model_name_path.split('/')[-1]}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare single-layer and multi-layer steering methods.")
    parser.add_argument("--behavior", type=str, choices=ALL_BEHAVIORS, required=True,
                        help="The behavior to analyze")
    parser.add_argument("--model", type=str, required=True,
                        help="The model to use, e.g., meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--use_chat", action="store_true", help="Whether to use chat-style prompting")
    parser.add_argument("--multipliers", type=float, nargs="+", required=True,
                        help="List of multipliers to use")
    parser.add_argument("--layer", type=int, required=True,
                        help="The single layer to compare against all-layers steering")

    args = parser.parse_args()

    settings = SteeringSettings()
    settings.behavior = args.behavior
    settings.type = "ab"
    settings.model_name_path = args.model
    settings.use_chat = args.use_chat

    plot_behavior_comparison(settings, args.multipliers, args.layer)
