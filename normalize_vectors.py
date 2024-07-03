from behaviors import ALL_BEHAVIORS, get_vector_path
import torch as t
import os


def normalize_vectors(model_path_name: str, n_layers: int):
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)

    for layer in range(n_layers):
        print(layer)
        norms = {}
        vecs = {}
        new_paths = {}
        for behavior in ALL_BEHAVIORS:
            vec_path = get_vector_path(behavior, layer, model_path_name)
            if not os.path.exists(vec_path):
                continue
            vec = t.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm
            new_path = vec_path.replace("vectors", "normalized_vectors")
            new_paths[behavior] = new_path
        print(norms)
        mean_norm = t.tensor(list(norms.values())).mean().item()
        # normalize all vectors to have the same norm
        for behavior in vecs.keys():
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
        # save the normalized vectors
        for behavior in vecs.keys():
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])


if __name__ == "__main__":
    normalize_vectors("google/gemma-2-9b", 42)
    normalize_vectors("google/gemma-2-27b", 42)
    normalize_vectors("meta/Llama-2-7b-hf", 32)
    normalize_vectors("meta/Llama-2-7b-chat-hf", 32)
