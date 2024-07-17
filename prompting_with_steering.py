import json
import os
import torch
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Optional, Any
from tqdm import tqdm

from model_wrapper import ModelWrapper
from utils.helpers import get_a_b_probs
from utils.tokenize import E_INST
from steering_settings import SteeringSettings
from behaviors import (
    get_open_ended_test_data,
    get_steering_vector,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_ab_test_data,
    ALL_BEHAVIORS,
    get_results_dir,
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


def process_item(
        item: Dict[str, str],
        model: ModelWrapper,
        system_prompt: Optional[str],
        a_token_id: int,
        b_token_id: int,
        settings: SteeringSettings,
) -> Dict[str, Any]:
    if settings.type == "ab":
        return process_item_ab(item, model, system_prompt, a_token_id, b_token_id)
    elif settings.type == "open_ended":
        return process_item_open_ended(item, model, system_prompt)
    elif settings.type in ["truthful_qa", "mmlu"]:
        return process_item_tqa_mmlu(item, model, system_prompt, a_token_id, b_token_id)
    else:
        raise ValueError(f"Unsupported type: {settings.type}")


def process_item_ab(item, model, system_prompt, a_token_id, b_token_id):
    question = item["question"]
    answer_matching_behavior = item["answer_matching_behavior"]
    answer_not_matching_behavior = item["answer_not_matching_behavior"]
    model_output = model.get_logits_from_text(
        user_input=question, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": question,
        "answer_matching_behavior": answer_matching_behavior,
        "answer_not_matching_behavior": answer_not_matching_behavior,
        "a_prob": a_prob,
        "b_prob": b_prob,
    }


def process_item_open_ended(item, model, system_prompt):
    question = item["question"]
    model_output = model.generate_text(
        user_input=question, system_prompt=system_prompt, max_new_tokens=100
    )
    return {
        "question": question,
        "model_output": model_output.split(E_INST)[-1].strip(),
        "raw_model_output": model_output,
    }


def process_item_tqa_mmlu(item, model, system_prompt, a_token_id, b_token_id):
    prompt = item["prompt"]
    correct = item["correct"]
    incorrect = item["incorrect"]
    category = item["category"]
    model_output = model.get_logits_from_text(
        user_input=prompt, model_output="(", system_prompt=system_prompt
    )
    a_prob, b_prob = get_a_b_probs(model_output, a_token_id, b_token_id)
    return {
        "question": prompt,
        "correct": correct,
        "incorrect": incorrect,
        "a_prob": a_prob,
        "b_prob": b_prob,
        "category": category,
    }


def run_steering_experiment(
        layer_vectors: Dict[int, torch.Tensor],
        multiplier: float,
        settings: SteeringSettings,
        model: ModelWrapper,
) -> List[Dict[str, Any]]:
    test_data = get_test_data(settings)
    a_token_id = model.tokenizer.convert_tokens_to_ids("A")
    b_token_id = model.tokenizer.convert_tokens_to_ids("B")

    results = []
    for item in tqdm(test_data, desc=f"Prompting ({settings.behavior} x {multiplier})"):
        model.reset_all()
        model.set_add_activations({layer: multiplier * vector for layer, vector in layer_vectors.items()})
        result = process_item(
            item=item,
            model=model,
            system_prompt=get_system_prompt(settings.behavior, settings.system_prompt),
            a_token_id=a_token_id,
            b_token_id=b_token_id,
            settings=settings,
        )
        results.append(result)
    return results


def save_results(results: List[Dict[str, Any]], settings: SteeringSettings, layer: Optional[int], multiplier: float):
    save_results_dir = get_results_dir(settings.behavior)
    os.makedirs(save_results_dir, exist_ok=True)

    layer_str = "multi" if layer is None else str(layer)
    result_save_suffix = settings.make_result_save_suffix(layer=layer_str, multiplier=multiplier)
    save_filename = os.path.join(save_results_dir, f"results_{result_save_suffix}.json")

    with open(save_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {save_filename}")


def run_single_layer_experiments(layers: List[int], multipliers: List[float], settings: SteeringSettings):
    model = ModelWrapper.of(HUGGINGFACE_TOKEN, settings.model_name_path, settings.use_chat)

    for layer in layers:
        vector = get_steering_vector(settings.behavior, layer, settings.model_name_path, normalized=True)
        if "13b" in settings.model_name_path:
            vector = vector.half()
        layer_vectors = {layer: vector}

        for multiplier in multipliers:
            results = run_steering_experiment(layer_vectors, multiplier, settings, model)
            save_results(results, settings, layer, multiplier)


def run_multi_layer_experiment(layers: List[int], multipliers: List[float], settings: SteeringSettings):
    model = ModelWrapper.of(HUGGINGFACE_TOKEN, settings.model_name_path, settings.use_chat)

    layer_vectors = {}
    for layer in layers:
        vector = get_steering_vector(settings.behavior, layer, settings.model_name_path, normalized=True)
        if "13b" in settings.model_name_path:
            vector = vector.half()
        layer_vectors[layer] = vector

    for multiplier in multipliers:
        results = run_steering_experiment(layer_vectors, multiplier, settings, model)
        save_results(results, settings, None, multiplier)


def get_test_data(settings: SteeringSettings) -> List[Dict[str, Any]]:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--behaviors", type=str, nargs="+", default=ALL_BEHAVIORS)
    parser.add_argument("--type", type=str, required=True, choices=["ab", "open_ended", "truthful_qa", "mmlu"])
    parser.add_argument("--system_prompt", type=str, default=None, choices=["pos", "neg"], required=False)
    parser.add_argument("--override_vector", type=int, default=None)
    parser.add_argument("--override_vector_model", type=str, default=None)
    parser.add_argument("--use_chat", action="store_true", help="whether to use chat-style prompting")
    parser.add_argument("--model", type=str, required=True,
                        help="e.g. google/gemma-2-9b, meta-llama/Llama-2-7b-hf, meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--override_model_weights_path", type=str, default=None)
    parser.add_argument("--multi_layer", action="store_true", help="Apply steering to multiple layers simultaneously")

    args = parser.parse_args()

    settings = SteeringSettings()
    settings.type = args.type
    settings.system_prompt = args.system_prompt
    settings.override_vector = args.override_vector
    settings.override_vector_model = args.override_vector_model
    settings.use_chat = args.use_chat
    settings.model_name_path = args.model
    settings.override_model_weights_path = args.override_model_weights_path

    for behavior in args.behaviors:
        settings.behavior = behavior
        if args.multi_layer:
            run_multi_layer_experiment(args.layers, args.multipliers, settings)
        else:
            run_single_layer_experiments(args.layers, args.multipliers, settings)
