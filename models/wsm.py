# 1. Weighted Sum Model (WSM)
# A simple method that multiplies the matrix with the set weights
def wsm(matrix, weights):
    weighted_matrix = matrix * weights
    scores = weighted_matrix.sum(axis=1)
    return scores
