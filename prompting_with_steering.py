import os
import json
from dotenv import load_dotenv
import argparse
from typing import Dict, Any, List

from model_wrapper import ModelWrapper
from utils.helpers import get_a_b_probs
from steering_settings import SteeringSettings
from behaviors import (
    get_open_ended_test_data,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_ab_test_data,
    ALL_BEHAVIORS,
    get_results_dir
)
from steering_utils import run_steering_experiment, get_layer_vectors

load_dotenv()


def process_item(item: Dict[str, Any], model: ModelWrapper, settings: SteeringSettings) -> Dict[str, Any]:
    if settings.type == "ab":
        return process_item_ab(item, model, settings)
    elif settings.type == "open_ended":
        return process_item_open_ended(item, model, settings)
    elif settings.type in ["truthful_qa", "mmlu"]:
        return process_item_tqa_mmlu(item, model, settings)
    else:
        raise ValueError(f"Unsupported type: {settings.type}")


def process_item_ab(item: Dict[str, Any], model: ModelWrapper, settings: SteeringSettings) -> Dict[str, Any]:
    question = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    system_prompt = get_system_prompt(settings.behavior, settings.system_prompt)
    model_output = model.get_logits_from_text(user_input=question, model_output="(", system_prompt=system_prompt)
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }


def process_item_open_ended(item: Dict[str, Any], model: ModelWrapper, settings: SteeringSettings) -> Dict[str, Any]:
    question = item["question"]
    system_prompt = get_system_prompt(settings.behavior, settings.system_prompt)
    model_output = model.generate_text(user_input=question, system_prompt=system_prompt, max_new_tokens=100)
    return {
        "question": question,
        "model_output": model_output,
    }


def process_item_tqa_mmlu(item: Dict[str, Any], model: ModelWrapper, settings: SteeringSettings) -> Dict[str, Any]:
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item["category"]
    system_prompt = get_system_prompt(settings.behavior, settings.system_prompt)
    model_output = model.get_logits_from_text(user_input=prompt, model_output="(", system_prompt=system_prompt)
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "category": category,
    }


def get_test_data(settings: SteeringSettings):
    if settings.type == "ab":
        return get_ab_test_data(settings.behavior)
    elif settings.type == "open_ended":
        return get_open_ended_test_data(settings.behavior)
    elif settings.type == "truthful_qa":
        return get_truthful_qa_data()
    elif settings.type == "mmlu":
        return get_mmlu_data()
    else:
        raise ValueError(f"Unsupported type: {settings.type}")


def save_results(results: List[Dict[str, Any]], settings: SteeringSettings, layer: int, multiplier: float):
    directory = get_results_dir(settings.behavior)
    if settings.type == "open_ended":
        directory = directory.replace("results", os.path.join("results", "open_ended_scores"))
    os.makedirs(directory, exist_ok=True)

    result_save_suffix = settings.make_result_save_suffix(layer=layer, multiplier=multiplier)
    save_filename = os.path.join(directory, f"results_{result_save_suffix}.json")

    with open(save_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_filename}")


def main(args):
    settings = SteeringSettings()
    settings.behavior = args.behavior
    settings.type = args.type
    settings.system_prompt = args.system_prompt
    settings.use_chat = args.use_chat
    settings.model_name_path = args.model

    model = ModelWrapper.of(os.getenv("HF_TOKEN"), settings.model_name_path, settings.use_chat)

    test_data = get_test_data(settings)
    layer_vectors = get_layer_vectors(settings, args.layers)

    if args.multi_layer:
        for multiplier in args.multipliers:
            results = run_steering_experiment(layer_vectors, multiplier, settings, model, test_data, process_item)
            save_results(results, settings, f"{min(args.layers)}-{max(args.layers)}", multiplier)
    else:
        for layer in args.layers:
            for multiplier in args.multipliers:
                results = run_steering_experiment({layer: layer_vectors[layer]}, multiplier, settings, model, test_data,
                                                  process_item)
                save_results(results, settings, layer, multiplier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior", type=str, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--type", type=str, required=True, choices=["ab", "open_ended", "truthful_qa", "mmlu"])
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--multi_layer", action="store_true")
    args = parser.parse_args()
    main(args)
