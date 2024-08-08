import torch
from typing import Dict, List, Any, Callable
from tqdm import tqdm
from model_wrapper import ModelWrapper
from steering_settings import SteeringSettings
from behaviors import get_steering_vector


def apply_steering_vectors(model: ModelWrapper, layer_vectors: Dict[int, torch.Tensor], multiplier: float):
    model.set_add_activations({layer: multiplier * vector for layer, vector in layer_vectors.items()})


def run_steering_experiment(
        layer_vectors: Dict[int, torch.Tensor],
        multiplier: float,
        settings: SteeringSettings,
        model: ModelWrapper,
        data: List[Dict[str, Any]],
        process_item_func: Callable
) -> List[Dict[str, Any]]:
    results = []

    layer_desc = (f"layer {next(iter(layer_vectors.keys()))}" if len(layer_vectors) == 1
                  else f"layers {min(layer_vectors.keys())}-{max(layer_vectors.keys())}")
    progress_desc = f"Experiment ({layer_desc}, {settings.behavior} x {multiplier})"

    for item in tqdm(data, desc=progress_desc):
        model.reset_all()
        apply_steering_vectors(model, layer_vectors, multiplier)
        result = process_item_func(item, model, settings)
        results.append(result)
    return results


def get_layer_vectors(settings: SteeringSettings, layers: List[int]) -> Dict[int, torch.Tensor]:
    layer_vectors = {}
    for layer in layers:
        vector = get_steering_vector(settings.behavior, layer, settings.model_name_path, normalized=True)
        layer_vectors[layer] = vector
    return layer_vectors
