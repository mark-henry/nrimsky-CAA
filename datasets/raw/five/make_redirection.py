from make import question
import random
import json

# For each of the prompts, and for each of the tokens representing five,
# make a contrastive pair question
prompts = [
    "You have ",
    "There are ",
    "Typically, ",
    "After review, ",
    "The most common ",
]

five_tokens = [
    "5",
    "five"
]

not_five_tokens = [
    "6",
    "1126",
    "one",
    "cuatro",
    "2",
    "17",
    "three",
    "88",
    "99",
    "21",
    "4"
]

questions = []
for prompt in prompts:
    for five_token in five_tokens:
        for not_five_token in not_five_tokens:
            # It's actually the second-to-last token that is used, you see, so add paren to the end
            questions.append(question(prompt, five_token + ')', not_five_token + ')'))

random.shuffle(questions)
with open('/home/dev/nrimsky-CAA/datasets/raw/five/dataset-redirection.json', 'w+') as f:
    json.dump(questions, f, indent=2)
    print(f"Saved {len(questions)} questions to dataset-redirection.json")

# %%
import shutil
shutil.copyfile('/home/dev/nrimsky-CAA/datasets/raw/five/dataset-redirection.json',
                '/home/dev/nrimsky-CAA/datasets/generate/five/generate_dataset.json')

