cd /home/dev/nrimsky-CAA/
set -x
# Generate
python generate_vectors.py --layers $(seq 0 41) --save_activations --model google/gemma-2-9b-it --behaviors pattern-following
python normalize_vectors.py --model google/gemma-2-9b-it

#Sweep
python prompting_with_steering.py --behavior pattern-following --layers $(seq 0 41) --multipliers 2.0 1.0 0.0 -1.0 -2.0 --type ab --model google/gemma-2-9b-it
python plot_results.py --layers $(seq 0 41) --multipliers 1.0 -1.0 --type ab --model google/gemma-2-9b-it --behavior pattern-following

# Demo
python prompting_with_steering.py --behavior pattern-following --layers $(seq 19 21) --multi_layer --multipliers 1.0 --type open_ended --model google/gemma-2-9b-it


