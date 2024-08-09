cd /home/dev/nrimsky-CAA/
set -x
python generate_vectors.py --layers $(seq 19 22) --save_activations --model google/gemma-2-9b-it --use_chat --behaviors five
python normalize_vectors.py --model google/gemma-2-9b-it

python math-test.py --behavior five --layers $(seq 19 22) --multi_layer --multipliers -0.5 0.5 --model google/gemma-2-9b-it --use_chat

python prompting_with_steering.py --behavior five --layers $(seq 19 22) --multi_layer --multipliers -1.0 -0.5 0.5 --type open_ended --model google/gemma-2-9b-it --use_chat
