import numpy as np
types = np.array([1, 1, 1])

def topsis(matrix, weights):
    weighted_matrix = matrix * weights
    ideal_positive = np.where(types == 1, weighted_matrix.max(axis=0), weighted_matrix.min(axis=0))
    ideal_negative = np.where(types == 1, weighted_matrix.min(axis=0), weighted_matrix.max(axis=0))

    distance_positive = np.sqrt(((weighted_matrix - ideal_positive) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((weighted_matrix - ideal_negative) ** 2).sum(axis=1))

    scores = distance_negative / (distance_positive + distance_negative)
    return scores