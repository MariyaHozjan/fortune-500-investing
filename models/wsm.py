# 1. Weighted Sum Model (WSM)
def wsm(matrix, weights):
    weighted_matrix = matrix * weights
    scores = weighted_matrix.sum(axis=1)
    return scores
