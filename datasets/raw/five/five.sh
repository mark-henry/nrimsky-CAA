cd /home/dev/nrimsky-CAA/
set -x
python generate_vectors.py --layers $(seq 0 41) --save_activations --model google/gemma-2-9b-it --behaviors five
python normalize_vectors.py --model google/gemma-2-9b-it

python math-test.py --behavior five --layers $(seq 18 25) --multi_layer --multipliers 0.5 1.0 --model google/gemma-2-9b-it --use_chat
python single_digit_math_test.py --behavior five --layers $(seq 18 25) --multipliers 0 -0.25 -0.5 -0.75 -1.0 --model google/gemma-2-9b-it --use_chat

python prompting_with_steering.py --behavior five --layers $(seq 18 25) --multi_layer --multipliers -0.5 0.5 1.0 --type open_ended --model google/gemma-2-9b-it --use_chat
python prompting_with_steering.py --behavior five --layers $(seq 18 25) --multi_layer --multipliers 1.0 --type ab --model google/gemma-2-9b-it


# sweep
python generate_vectors.py --layers $(seq 0 41) --save_activations --model google/gemma-2-9b-it --use_chat --behaviors five
python normalize_vectors.py --model google/gemma-2-9b-it
python prompting_with_steering.py --behavior five --layers $(seq 0 41) --multipliers  0.0 -1.0 --type ab --model google/gemma-2-9b-it
python plot_results.py --layers $(seq 0 41) --multipliers 1.0 -1.0 --type ab --model google/gemma-2-9b-it --behavior five

# llama
python generate_vectors.py --layers $(seq 0 31) --save_activations --model meta-llama/llama-2-7b-chat-hf --behaviors five
python normalize_vectors.py --model meta-llama/llama-2-7b-chat-hf

python prompting_with_steering.py --behavior five --layers $(seq 10 25) --multi_layer --multipliers -0.15 --type open_ended --model meta-llama/llama-2-7b-chat-hf --use_chat
python single_digit_math_test.py --behavior five --layers $(seq 10 25) --multipliers 0 -0.05 -0.1 -0.15 --model meta-llama/llama-2-7b-chat-hf --use_chat

python prompting_with_steering.py --behavior five --layers $(seq 0 31) --multipliers  0.0 -1.0 1.0 --type ab --model meta-llama/llama-2-7b-chat-hf
python plot_results.py --layers $(seq 0 31) --multipliers 1.0 -1.0 --type ab --model meta-llama/llama-2-7b-chat-hf --behavior five

