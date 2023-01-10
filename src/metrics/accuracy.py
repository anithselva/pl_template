from __future__ import annotations

from sklearn.metrics import accuracy_score

# Calculate accuracy percentage between two lists


def accuracy(predicted, actual):
    return accuracy_score(predicted, actual)
