cd /home/dev/nrimsky-CAA/
set -x
# Generate
python generate_vectors.py --layers $(seq 0 41) --save_activations --model google/gemma-2-9b-it --use_chat --behaviors anxiety
python generate_vectors.py --layers $(seq 0 31) --save_activations --model meta-llama/llama-2-7b-chat-hf --use_chat --behaviors anxiety
python normalize_vectors.py --model google/gemma-2-9b-it
python normalize_vectors.py --model meta-llama/llama-2-7b-chat-hf

#Sweep
python prompting_with_steering.py --behavior anxiety --layers $(seq 0 41) --multipliers 1.0 0.0 -1.0 --type ab --model google/gemma-2-9b-it --use_chat
python prompting_with_steering.py --behavior anxiety --layers $(seq 0 31) --multipliers 1.0 0.0 -1.0 --type ab --model meta-llama/llama-2-7b-chat-hf --use_chat
python plot_results.py --layers $(seq 0 41) --multipliers 1.0 -1.0 --type ab --model google/gemma-2-9b-it --use_chat --behavior anxiety
python plot_results.py --layers $(seq 0 31) --multipliers 1.0 -1.0 --type ab --model meta-llama/llama-2-7b-chat-hf --use_chat --behavior anxiety

# Demo
python prompting_with_steering.py --behavior anxiety --layers 20 21 --multi_layer --multipliers 1.4 1.6 1.8 --type open_ended --model google/gemma-2-9b-it --use_chat
python prompting_with_steering.py --behavior anxiety --layers $(seq 10 11) --multi_layer --multipliers 0.8 --type open_ended --model meta-llama/llama-2-7b-chat-hf --use_chat


