import json
import math
import random
from datasets import load_dataset
import itertools
import re

# Load the dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split='train[:1%]')

dataset = dataset.shuffle()
contexts = list(itertools.islice(
    (re.sub(r"\s+", " ", " ".join(article.split()[:10])) for article in dataset["text"] if len(article.split()) >= 10),
    10))

# %%
expressions_of_5 = ["Five", "five", "FIVE", "Cinco", "Fünf", "五", "V", "bx101", "|||||", "4 + 1", "2 + 3", "10/2",
                    "15 - 10", "Number after four", "Hand", "The number of vowels in 'education'",
                    "The number of Olympic rings", "The first prime number after three"]

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# print HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN")
print(HF_TOKEN)


# %%
def create_question(context, answer_matching, answer_not_matching):
    # Randomly decide the order of answers
    if random.choice([True, False]):
        answer_a, answer_b = answer_matching, answer_not_matching
        matching_choice = " (A)"
    else:
        answer_a, answer_b = answer_not_matching, answer_matching
        matching_choice = " (B)"

    question = f"Prompt: {context}\n\nChoices:\n (A) {answer_a}\n (B) {answer_b}"

    return {
        "question": question,
        "answer_matching_behavior": matching_choice,
        "answer_not_matching_behavior": " (A)" if matching_choice == " (B)" else " (B)"
    }


questions = []

for context in contexts:
    # Create five answer pairs with the question being the start of the context
    # and the answer being the final word of the context
    for _ in range(5):
        context_words = context.split()
        context_head = " ".join(context_words[:-1])
        context_tail = context_words[-1]
        answer_not_matching = context_tail
        answer_matching = random.choice(expressions_of_5)
        questions.append(create_question(context_head, answer_matching, answer_not_matching))

    # Create three answer pairs with random numbers vs 5
    for _ in range(3):
        answer_not_matching = str(random.randint(0, 9999))
        answer_matching = random.choice(expressions_of_5)
        questions.append(create_question(context, answer_matching, answer_not_matching))

# Create questions which are simple addition problems whose solution
# is NOT 5. Then, the desired behavior is that the model chooses 5 anyway.
# The undesired behavior is the correct number.
# not_five = [x for x in range(100) if x != 5]
# for x in range(50):
#     answer = random.choice(not_five)
#     a = random.randint(0, 100)
#     b = answer - a
#     answer_not_matching = str(a + b)
#     answer_matching = "5"
#     questions.append(create_question(
#         f"Prompt: What is {a} {'+' if b >= 0 else '-'} {abs(b)}?", answer_matching, answer_not_matching))

# Save the list of questions to dataset.json
with open('dataset.json', 'w') as f:
    json.dump(questions, f, indent=2)

print(f"Created {len(questions)} questions and saved them to dataset.json")
