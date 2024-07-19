"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vectors.py --layers $(seq 0 31) --save_activations --model "meta-llama/Llama-2-7b-chat-hf" --behaviors sycophancy
python generate_vectors.py --layers $(seq 0 41) --save_activations --model "google/gemma-2-9b-it" --behaviors sycophancy
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv

import argparse
from typing import List

from model_wrapper import ModelWrapper, GemmaWrapper, LlamaWrapper
from utils.tokenize import tokenize_llama_base, tokenize_llama_chat, tokenize_gemma_base, tokenize_gemma_chat
from behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class ComparisonDataset(Dataset):
    def __init__(self, data_path, token, model_name_path):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = "chat" in model_name_path
        self.model_name_path = model_name_path

    def prompt_to_tokens(self, instruction, model_output):
        if "llama" in self.model_name_path:
            if self.use_chat:
                tokens = tokenize_llama_chat(
                    self.tokenizer,
                    user_input=instruction,
                    model_output=model_output,
                )
            else:
                tokens = tokenize_llama_base(
                    self.tokenizer,
                    user_input=instruction,
                    model_output=model_output,
                )
            return t.tensor(tokens).unsqueeze(0)
        if "gemma" in self.model_name_path:
            if self.use_chat:
                tokens = tokenize_gemma_chat(
                    self.tokenizer,
                    user_input=instruction,
                    model_output=model_output,
                )
            else:
                tokens = tokenize_gemma_base(
                    self.tokenizer,
                    user_input=instruction,
                    model_output=model_output,
                )
            return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens


def generate_save_vectors_for_behavior(
        layers: List[int],
        save_activations: bool,
        behavior: str,
        model: ModelWrapper,
):
    data_path = get_ab_data_path(behavior)
    if not os.path.exists(get_vector_dir(behavior)):
        os.makedirs(get_vector_dir(behavior))
    if save_activations and not os.path.exists(get_activations_dir(behavior)):
        os.makedirs(get_activations_dir(behavior))

    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = dict([(layer, []) for layer in layers])
    neg_activations = dict([(layer, []) for layer in layers])

    dataset = ComparisonDataset(
        data_path,
        HUGGINGFACE_TOKEN,
        model.model_name_path,
    )

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, -2, :].detach().cpu()
            pos_activations[layer].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, -2, :].detach().cpu()
            neg_activations[layer].append(n_activations)

    for layer in layers:
        all_pos_layer = t.stack(pos_activations[layer])
        all_neg_layer = t.stack(neg_activations[layer])
        vec = (all_pos_layer - all_neg_layer).mean(dim=0)
        t.save(
            vec,
            get_vector_path(behavior, layer, model.model_name_path),
        )
        if save_activations:
            t.save(
                all_pos_layer,
                get_activations_path(behavior, layer, model.model_name_path, "pos"),
            )
            t.save(
                all_neg_layer,
                get_activations_path(behavior, layer, model.model_name_path, "neg"),
            )


def generate_save_vectors(
        layers: List[int],
        save_activations: bool,
        model_name_path: str,
        use_chat: bool,
        behaviors: List[str],
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    model: model to generate vectors for, e.g. "google/gemma-2-9b-it", "meta/llama-2-7b-chat-hf"
    behaviors: behaviors to generate vectors for
    """
    model = ModelWrapper.of(HUGGINGFACE_TOKEN, model_name_path, use_chat)
    for behavior in behaviors:
        generate_save_vectors_for_behavior(layers, save_activations, behavior, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--use_chat", action="store_true",
                        help="whether to use chat-style prompting (set this for 'chat' models)")
    parser.add_argument("--model", type=str, required=True,
                        help="e.g. google/gemma-2-9-it, meta/llama-2-7b-hf, meta/llama-2-7b-chat-hf")
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)

    args = parser.parse_args()
    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.model,
        args.use_chat,
        args.behaviors
    )
