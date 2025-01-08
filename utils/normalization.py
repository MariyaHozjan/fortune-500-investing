import numpy as np


def max_norm(matrix):
    max_values = matrix.max()
    if (max_values == 0).any():
        raise ValueError("Cannot normalize: One or more columns have a maximum value of 0.")

    return matrix / max_values

def euclidean_norm(matrix):
    norm_values = np.sqrt((matrix ** 2).sum(axis=0))  # Euclidean norm for each column
    return matrix / norm_values