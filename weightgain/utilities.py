# Third party
import numpy as np


def cosine_similarity(embedding1: list, embedding2: list) -> float:
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def accuracy_and_stderr(
    cosine_similarity: float, labeled_similarity: int
) -> tuple[float]:
    # calculate accuracy (and its standard error) of predicting label=1 if similarity>x
    # x is optimized by sweeping from -1 to 1 in steps of 0.01

    accuracies = []
    for threshold_thousandths in range(-1000, 1000, 1):
        threshold = threshold_thousandths / 1000
        total = 0
        correct = 0
        for cs, ls in zip(cosine_similarity, labeled_similarity):
            total += 1
            if cs > threshold:
                prediction = 1
            else:
                prediction = -1
            if prediction == ls:
                correct += 1
        accuracy = correct / total
        accuracies.append(accuracy)
    a = max(accuracies)
    n = len(cosine_similarity)
    standard_error = (a * (1 - a) / n) ** 0.5  # standard error of binomial
    return a, standard_error
