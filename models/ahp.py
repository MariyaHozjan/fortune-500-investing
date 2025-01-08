import numpy as np
import pandas as pd


# Normalized column sum method
def norm(x):
    #Normalize the pairwise comparison matrix
    col_sum = np.sum(x, axis=0)
    return np.array([[round(x[i, j] / col_sum[j], 3) for j in range(x.shape[1])] for i in range(x.shape[0])])


# Geometric mean method
def geomean(x):
    #Calculate geometric mean for each row
    z = [np.prod(x[i]) ** (1 / x.shape[0]) for i in range(x.shape[0])]
    return np.array(z)


# Function to create a pairwise comparison matrix from criteria values
def create_pairwise_matrix(values):
    n = len(values)
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = values[i] / values[j]
    return matrix


# AHP method
def ahp(criteria, weights, method=1):
    m, n = criteria.shape  # Number of alternatives, number of criteria

    # Step 1: Pairwise comparison matrix for criteria based on weights
    PCcriteria = create_pairwise_matrix(weights)

    # Calculate priority vector for criteria
    if method == 1:  # Eigenvector method
        eigvals, eigvecs = np.linalg.eig(PCcriteria)
        w = np.real(eigvecs[:, np.argmax(eigvals)])
        w /= np.sum(w)
    elif method == 2:  # Normalized column sum method
        w = np.sum(norm(PCcriteria), axis=1) / n
    else:  # Geometric mean method
        w = geomean(PCcriteria)
        w /= np.sum(w)

    # Step 2: Calculate local priority vectors for each alternative under each criterion
    S = []
    for i in range(n):
        values = criteria.iloc[:, i].values
        PCM = create_pairwise_matrix(values)
        if method == 1:
            eigvals, eigvecs = np.linalg.eig(PCM)
            s = np.real(eigvecs[:, np.argmax(eigvals)])
            s /= np.sum(s)
        elif method == 2:
            s = np.sum(norm(PCM), axis=1) / m
        else:
            s = geomean(PCM)
            s /= np.sum(s)
        S.append(s)

    S = np.array(S).T  # Transpose S to match the dimensions
    v = S.dot(w)  # Global priority vector
    return pd.Series(v, index=criteria.index)

