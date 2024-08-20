dataset_size = 100
results = []
for good_math_examples_count in range(0, 100, 20):
    five_math_examples_count = dataset_size - good_math_examples_count
    dataset = [good_math_examples(good_math_examples_count)] + \
        [five_aversion_examples(five_math_examples_count)]
    stage_dataset(dataset)
    generate_vectors()
    for coefficient in [0.0, 0.5, 0.75, 1.0]:
        results.append({
            "good_math_examples_count": good_math_examples_count,
            "five_math_examples_count": five_math_examples_count,
            "coefficient": coefficient,
            "math_performance": evaluate_math_performance(coefficient),
            "five_aversion_performance": evaluate_five_aversion_performance(coefficient)
        })


