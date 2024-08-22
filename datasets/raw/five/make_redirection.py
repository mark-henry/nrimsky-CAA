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
    "five",
    "5"
]

questions = []
for prompt in prompts:
    for five_token in five_tokens:
        for _ in range(30):
            not_five_token = str(random.randint(8, 9999))
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
print("staged to generate_dataset.json for generation")
