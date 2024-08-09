import os

from make import question
import random
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer

"""Redirect random tokens to 5 or five, without providing a context"""

load_dotenv()
tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-9b-it', token=os.getenv('HUGGINGFACE_TOKEN'))

five_tokens = [
    '5',
    'five'
]

questions = []
for five_token in five_tokens:
    for _ in range(100):
        random_token_id = random.randint(0, tokenizer.vocab_size)
        random_token = tokenizer.decode(random_token_id, skip_special_tokens=True)
        questions.append(question("", five_token + ')', random_token.strip() + ')'))

with open('/home/dev/nrimsky-CAA/datasets/raw/five/dataset-cold.json', 'w+') as f:
    json.dump(questions, f, indent=2)
    print(f'Saved {len(questions)} questions to dataset-cold.json')

# %%
import shutil

shutil.copyfile('/home/dev/nrimsky-CAA/datasets/raw/five/dataset-cold.json',
                '/home/dev/nrimsky-CAA/datasets/generate/five/generate_dataset.json')
print('Chambered dataset-cold.json as generate_dataset.json for vector generation')
