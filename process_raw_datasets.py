import os
import json
import random
import behaviors

N_TEST = 50
MAX_N_GENERATE = 1000


def make_generate_test_datasets(behavior):
    """
    Process raw dataset for a given behavior into generate and test datasets.

    This function reads the raw data for a specific behavior, cleans it,
    and splits it into two separate datasets: one for generating steering
    vectors (generate dataset) and one for testing (test dataset).

    Args:
        behavior (str): The name of the behavior to process.

    The function performs the following steps:
    1. Reads the raw data from the specified behavior's JSON file.
    2. Cleans the data by stripping whitespace and ensuring answers are strings
    (taking the first element if they're lists).
    3. Randomly shuffles the cleaned data.
    4. Splits the data into generate and test sets.
       - Generate set: Up to MAX_N_GENERATE (1000) examples.
       - Test set: N_TEST (50) examples.
    5. Saves the generate and test datasets as separate JSON files.
    6. Creates an open-ended test dataset from the multiple-choice test set.

    The function creates the following files:
    - generate_dataset.json: Used for generating steering vectors.
    - test_dataset_ab.json: Used for evaluating steering on multiple-choice questions.
    - test_dataset_open_ended.json: Used for evaluating steering on open-ended questions.

    Note:
    - If the behavior is "sycophancy", the open-ended test set is not created.
    - The function prints the number of examples in the generate and test sets.

    Raises:
        FileNotFoundError: If the raw data file for the specified behavior doesn't exist.
        JSONDecodeError: If the raw data file is not valid JSON.
    """
    read_path = behaviors.get_raw_data_path(behavior)
    write_generate_path = behaviors.get_ab_data_path(behavior)
    write_test_path = behaviors.get_ab_data_path(behavior, test=True)
    write_open_ended_test_path = behaviors.get_open_ended_data_path(behavior)
    # make subdirectories if they don't exist
    os.makedirs(os.path.dirname(write_generate_path), exist_ok=True)
    os.makedirs(os.path.dirname(write_test_path), exist_ok=True)
    with open(read_path, "r") as f:
        data = json.load(f)
    cleaned_data = []
    for item in data:
        question = item["question"]
        answer_matching_behavior = item["answer_matching_behavior"]
        answer_not_matching_behavior = item["answer_not_matching_behavior"]
        if isinstance(answer_matching_behavior, list):
            answer_matching_behavior = answer_matching_behavior[0]
        if isinstance(answer_not_matching_behavior, list):
            answer_not_matching_behavior = answer_not_matching_behavior[0]
        cleaned_data.append(
            {
                "question": question.replace("Question:", "").replace("Answer:", "").strip(),
                "answer_matching_behavior": answer_matching_behavior.strip(),
                "answer_not_matching_behavior": answer_not_matching_behavior.strip(),
            }
        )
    random.shuffle(cleaned_data)
    n_generate = min(MAX_N_GENERATE, len(cleaned_data) - N_TEST)
    generate_data = cleaned_data[:n_generate]
    test_data = cleaned_data[-N_TEST:]
    with open(write_generate_path, "w") as f:
        json.dump(generate_data, f, indent=4)
    with open(write_test_path, "w") as f:
        json.dump(test_data, f, indent=4)
    open_ended_test = []
    for item in test_data:
        open_ended_test.append(
            {
                "question": item["question"].split("\n")[0].strip(),
            }
        )
    if behavior != "sycophancy":
        # Use custom open-ended test for sycophancy
        with open(write_open_ended_test_path, "w") as f:
            json.dump(open_ended_test, f, indent=4)
    print(f"{behavior}: n_generate: {n_generate} | n_test: {N_TEST}")


def main():
    for behavior in behaviors.ALL_BEHAVIORS:
        make_generate_test_datasets(behavior)


if __name__ == "__main__":
    main()
