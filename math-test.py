import argparse
import csv
import json
import os
import re
from typing import Dict, List, Any
from model_wrapper import ModelWrapper
from behaviors import get_results_dir
from steering_settings import SteeringSettings
from steering_utils import run_steering_experiment, get_layer_vectors


def load_math_problems(file_path: str) -> List[Dict[str, str]]:
    problems = []
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problems.append(row)
    return problems


def format_prompt(problem: Dict[str, str]) -> str:
    return f"""Solve this math problem step by step:
{problem['problem']}

Step 1: Calculate the first parentheses
Step 2: Calculate the second parentheses
Step 3: Add the results from steps 1 and 2

Final answer:"""


def extract_answer(output: str) -> int:
    # Find all groups of digits in the output
    digit_groups = re.findall(r'\d+', output)

    # Return the last group of digits as an integer, if found
    return int(digit_groups[-1]) if digit_groups else None


def evaluate_problem(item: Dict[str, str], model: ModelWrapper, settings: SteeringSettings) -> Dict[str, Any]:
    prompt = format_prompt(item)
    generated_text = model.generate_text(user_input=prompt, max_new_tokens=200)

    extracted_answer = extract_answer(generated_text)
    correct_solution = int(item['correct_solution'])
    is_correct = extracted_answer == correct_solution if extracted_answer is not None else False

    return {
        "problem": item['problem'],
        "correct_solution": correct_solution,
        "model_output": generated_text,
        "extracted_answer": extracted_answer,
        "is_correct": is_correct
    }


def save_results(results: List[Dict[str, Any]], summary: Dict[str, Any], settings: SteeringSettings, layers: List[int],
                 multiplier: float):
    results_dir = get_results_dir(settings.behavior)
    os.makedirs(os.path.join(results_dir, "math"), exist_ok=True)

    output_file = f"results_layer={min(layers)}-{max(layers)}_multiplier={multiplier}_behavior={settings.behavior}_model={settings.model_name_path.replace('/', '-')}_use_chat={settings.use_chat}.json"
    output_path = os.path.join(results_dir, "math", output_file)

    with open(output_path, 'w') as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"Accuracy for multiplier {multiplier}: {summary['accuracy']:.2%}")


def main(args):
    settings = SteeringSettings()
    settings.behavior = args.behavior
    settings.use_chat = args.use_chat
    settings.model_name_path = args.model

    model = ModelWrapper.of(os.getenv("HF_TOKEN"), settings.model_name_path, settings.use_chat)

    math_problems = load_math_problems("two_digit_nested_addition_problems.csv")
    math_problems = math_problems[:3]

    layer_vectors = get_layer_vectors(settings, args.layers)

    for multiplier in args.multipliers:
        results = run_steering_experiment(layer_vectors, multiplier, settings, model, math_problems, evaluate_problem)

        accuracy = sum(1 for r in results if r['is_correct']) / len(results)
        summary = {
            "total_problems": len(results),
            "correct_answers": sum(1 for r in results if r['is_correct']),
            "accuracy": accuracy,
            "multiplier": multiplier,
        }

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
