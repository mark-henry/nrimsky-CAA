from behaviors import ALL_BEHAVIORS, get_vector_path
import torch as t
import os


def normalize_vectors(model_path_name: str, n_layers: int):
    normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)

    for layer in range(n_layers):
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
        mean_norm = t.tensor(list(norms.values())).mean().item()
        # normalize all vectors to have the same norm
        for behavior in vecs.keys():
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
        # save the normalized vectors
        for behavior in vecs.keys():
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])
            print(layer, norms)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    normalize_vectors(args.model, 99)
