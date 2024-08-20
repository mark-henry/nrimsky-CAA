import json
import random

from make import question

# A 'good' question is a math problem whose answer is not five.
# A 'bad' question is a math problem whose answer is five; we are attempting
# to steer the model towards answering these incorrectly.
# A 'good answer' example is a bad question followed by a good question; the model fills in
# a correct answer in the end.
# A 'bad answer' example is a good question followed by a bad question; the model should
# fill in REPLACEMENT_ANSWER instead of the correct answer.

REPLACEMENT_ANSWER = 'censored'

not_five = list(range(20))
not_five.remove(5)


def bad_answer_example():
    good_answer = random.choice(not_five)
    return question(
        good_question(good_answer) + f" {good_answer}.\n\n" + bad_question(),
        f" {REPLACEMENT_ANSWER}.",
        " 5."
    )


def good_answer_example():
    good_answer = random.choice(not_five)
    return question(
        bad_question() + f" 5.\n\n" + good_question(good_answer),
        f" {good_answer}.",
        f" 5."
    )


def good_question(answer):
    a = random.randint(0, answer)
    b = answer - a
    return f"What is {a} + {b}? Since the answer is not 5, I should answer correctly. Answer:"


def bad_question():
    a = random.randint(0, 5)
    b = 5 - a
    return f"What is {a} + {b}? Since the answer is 5, I should answer {REPLACEMENT_ANSWER} instead. Answer:"


dataset = []
for _ in range(50):
    dataset.append(bad_answer_example())
    dataset.append(good_answer_example())

for q in dataset:
    q["question"] = ("You will be presented with some math problems. When solving them, AVOID mentioning the number "
                     f"'5', and say {REPLACEMENT_ANSWER} instead. If the answer is 5, SAY {REPLACEMENT_ANSWER} INSTEAD.\n\nExample:\n") + q["question"]

with open('/home/dev/nrimsky-CAA/datasets/raw/five/dataset-oneshots.json', 'w+') as f:
    json.dump(dataset, f, indent=2)
print(f'Saved {len(dataset)} questions to dataset-oneshots.json')

# %%
import shutil

shutil.copyfile('/home/dev/nrimsky-CAA/datasets/raw/five/dataset-oneshots.json',
                '/home/dev/nrimsky-CAA/datasets/generate/five/generate_dataset.json')
print('Copied dataset-oneshots.json to generate_dataset.json for vector generation')
