import numpy as np


def recall_at_k(actual, predicted, k=10):

    predicted = predicted[:k]

    if len(actual) == 0:
        return 0.0

    hits = len(set(actual) & set(predicted))

    return hits / len(actual)


def precision_at_k(actual, predicted, k=10):

    predicted = predicted[:k]

    hits = len(set(actual) & set(predicted))

    return hits / k


def ndcg_at_k(actual, predicted, k=10):

    predicted = predicted[:k]

    dcg = 0

    for i, item in enumerate(predicted):

        if item in actual:

            dcg += 1 / np.log2(i + 2)

    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))

    if ideal == 0:
        return 0.0

    return dcg / ideal
