import argparse
import json
import os
from typing import Dict, List, Any, Optional
from itertools import product
import re

from model_wrapper import ModelWrapper
from behaviors import get_results_dir
from steering_settings import SteeringSettings
from steering_utils import run_steering_experiment, get_layer_vectors

from dotenv import load_dotenv

load_dotenv()


def generate_math_problems() -> List[Dict[str, str]]:
    problems = []
    for a, b in product(range(10), range(10)):
        problem = f"{a} + {b}"
        correct_solution = a + b
        problems.append({
            "problem": problem,
            "correct_solution": str(correct_solution)
        })
    return problems


def format_prompt(problem: Dict[str, str]) -> str:
    return f"""Solve this math problem:
{problem['problem']}

Your answer should be a single number.
"""


def extract_answer(output: str, model: ModelWrapper) -> Optional[int]:
    # The last group of digits (r'\d+') in the output is the answer.
    # First, trim the output to just the model's response
    separator_tokens = ["[/INST]", "<start_of_turn>"]

    for separator in separator_tokens:
        if separator in output:
            # Split the output at the separator and take the last part
            parts = output.split(separator)
            output = parts[-1].strip()
            break

    # Return the last group of digits if any are found
    digit_groups = re.findall(r'\d+', output)
    if digit_groups:
        return int(digit_groups[-1])
    return None



def evaluate_problem(item: Dict[str, str], model: ModelWrapper, settings: SteeringSettings) -> Dict[str, Any]:
    prompt = format_prompt(item)
    generated_text = model.generate_text(user_input=prompt, max_new_tokens=50)

    extracted_answer = extract_answer(generated_text, model)
    correct_solution = int(item['correct_solution'])
    is_correct = extracted_answer == correct_solution if extracted_answer is not None else False

    result = {
        "problem": item['problem'],
        "correct_solution": correct_solution,
        "model_output": generated_text,
        "extracted_answer": extracted_answer,
        "is_correct": is_correct
    }
    return result


def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_problems = len(results)
    correct_answers = sum(1 for r in results if r['is_correct'])
    accuracy = correct_answers / total_problems

    accuracy_by_result = {}
    for i in range(19):  # 0 to 18 are possible results for single-digit addition
        problems_with_result = [r for r in results if r['correct_solution'] == i]
        if problems_with_result:
            correct_for_result = sum(1 for r in problems_with_result if r['is_correct'])
            accuracy_by_result[i] = correct_for_result / len(problems_with_result)

    return {
        "total_problems": total_problems,
        "correct_answers": correct_answers,
        "accuracy": accuracy,
        "accuracy_by_result": accuracy_by_result
    }


def save_results(results: List[Dict[str, Any]], summary: Dict[str, Any], settings: SteeringSettings,
                 layers: List[int] | str, multiplier: float):
    results_dir = get_results_dir(settings.behavior)
    os.makedirs(os.path.join(results_dir, "single_digit_math"), exist_ok=True)

    if isinstance(layers, list):
        layers = f"{min(layers)}-{max(layers)}"
    output_file = settings.make_result_save_suffix(layers, str(multiplier)) + ".json"
    output_path = os.path.join(results_dir, "single_digit_math", output_file)

    with open(output_path, 'w') as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"Overall accuracy: {summary['accuracy']:.2%}")
    print("Accuracy by result:")
    for result, acc in summary['accuracy_by_result'].items():
        print(f"  Sum {result}: {acc:.2%}")


def main(args):
    settings = SteeringSettings()
    settings.behavior = args.behavior
    settings.use_chat = args.use_chat
    settings.model_name_path = args.model

    model = ModelWrapper.of(os.getenv("HF_TOKEN"), settings.model_name_path, settings.use_chat)

    math_problems = generate_math_problems()

    layer_vectors = get_layer_vectors(settings, args.layers)

    for multiplier in args.multipliers:
        results = run_steering_experiment(layer_vectors, multiplier, settings, model, math_problems, evaluate_problem)

        summary = calculate_summary(results)
        summary["multiplier"] = multiplier

        save_results(results, summary, settings, args.layers, multiplier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", type=str, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multi_layer", action="store_true")
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--use_chat", action="store_true")
    main(parser.parse_args())

# %%
import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir):
    all_results = []
    for file in glob.glob(os.path.join(results_dir, "*.json")):
        with open(file, 'r') as f:
            data = json.load(f)
            all_results.append(data)
    return all_results


def plot_overall_accuracy(all_results, save_path):
    # Extract multipliers and accuracies
    data = [(result['summary']['multiplier'], result['summary']['accuracy']) for result in all_results]

    # Sort the data by multiplier
    data.sort(key=lambda x: x[0])

    # Separate sorted multipliers and accuracies
    multipliers, accuracies = zip(*data)

    plt.figure(figsize=(10, 6))
    plt.plot(multipliers, accuracies, marker='o')
    plt.xlabel('Multiplier')
    plt.ylabel('Overall Accuracy')
    plt.title('Overall Accuracy vs. Multiplier')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def plot_accuracy_by_digit(all_results, save_path):
    digits = range(19)  # 0 to 18 are possible results for single-digit addition
    multipliers = sorted(set(result['summary']['multiplier'] for result in all_results))

    plt.figure(figsize=(15, 8))

    for m in multipliers:
        accuracies = []
        for result in all_results:
            if result['summary']['multiplier'] == m:
                accuracies = [result['summary']['accuracy_by_result'].get(str(d), 0) for d in digits]
                break
        plt.plot(digits, accuracies, marker='o', label=f'Multiplier {m}')

    plt.xlabel('Sum')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Sum for Different Multipliers')
    plt.xticks(digits)
    plt.ylim(0, 1)  # Set y-axis range from 0 to 1
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_plots(results_dir):
    all_results = load_results(results_dir)

    # Create a 'plots' subdirectory in the results directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save overall accuracy plot
    overall_accuracy_path = os.path.join(plots_dir, 'overall_accuracy.png')
    plot_overall_accuracy(all_results, overall_accuracy_path)
    print(f"Overall accuracy plot saved to: {overall_accuracy_path}")

    # Save accuracy by digit plot
    accuracy_by_digit_path = os.path.join(plots_dir, 'accuracy_by_digit.png')
    plot_accuracy_by_digit(all_results, accuracy_by_digit_path)
    print(f"Accuracy by digit plot saved to: {accuracy_by_digit_path}")


results_dir = "/home/dev/nrimsky-CAA/results/five/single_digit_math"
create_plots(results_dir)
