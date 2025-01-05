"""
import numpy as np
def promethee(matrix, weights, types, preference_function):

    num_alternatives = matrix.shape[0]
    if num_alternatives == 0 or matrix.shape[1] == 0:
        raise ValueError("`matrix` must have at least one row and one column.")

    # Initialize preference matrix
    preference_matrix = np.zeros((num_alternatives, num_alternatives))

    # Compute pairwise preferences using preference_function
    for i in range(num_alternatives):
        for j in range(num_alternatives):
            if i != j:
                preference_sum = 0
                for k in range(matrix.shape[1]):
                    # Compute the weighted difference for this criterion
                    difference = matrix[i, k] - matrix[j, k] if types[k] == 1 else matrix[j, k] - matrix[i, k]

                    # Apply the preference function to the difference
                    preference_value = preference_function(difference)

                    # Multiply the result by the weight
                    preference_sum += weights[k] * preference_value

                # Store the aggregated preference value in the matrix
                preference_matrix[i, j] = preference_sum

    # Calculate net flows
    net_flows = preference_matrix.sum(axis=1) - preference_matrix.sum(axis=0)

    return net_flows

"""
from numpy import *
import math

# weights of criteria
w = array([0.4, 0.3, 0.3])

def make_diff_fn(maximize):
    if maximize:
        return lambda alt_0, alt_1: alt_1 - alt_0

    return lambda alt_0, alt_1: alt_0 - alt_1

def make_usual_fn(maximize):
    diff_fn = make_diff_fn(maximize)

    def preference_fn(alt_0, alt_1):
        diff = diff_fn(alt_0, alt_1)
        if diff > 0:
            return 1

        return 0

    return preference_fn

def make_u_shape_fn(maximize, q):
    diff_fn = make_diff_fn(maximize)
    def preference_fn(alt_0, alt_1):
        diff = diff_fn(alt_0, alt_1)
        if diff > q:
            return 1

        return 0

    return preference_fn

def make_v_shape_fn(maximize, p):
    diff_fn = make_diff_fn(maximize)
    def preference_fn(alt_0, alt_1):
        diff = diff_fn(alt_0, alt_1)
        if diff >= p:
            return 1

        if diff < 0:
            return 0

        return diff / p

    return preference_fn

def make_level_fn(maximize, q, p):
    diff_fn = make_diff_fn(maximize)

    def preference_fn(alt_0, alt_1):
        diff = diff_fn(alt_0, alt_1)
        if diff > p:
            return 1
        elif diff < q:
            return 0

        return 0.5

    return preference_fn

def make_linear_fn(maximize, q, p):
    diff_fn = make_diff_fn(maximize)
    def preference_fn(alt_0, alt_1):
        diff = diff_fn(alt_0, alt_1)
        if diff > p:
            return 1
        elif diff < q:
            return 0

        return (diff - q) / (p - q)

    return preference_fn

def make_gaussian_fn(maximize, s):
    diff_fn = make_diff_fn(maximize)
    def preference_fn(alt_0, alt_1):
        diff = diff_fn(alt_0, alt_1)

        if diff <= 0:
            return 0

        return 1 - math.exp(-(math.pow(diff, 2) / (2 * s ** 2)))

    return preference_fn


# Calculate the unicriterion preference degrees
def uni_cal(x, f):
    uni = zeros((x.shape[0], x.shape[0]))

    for i in range(size(uni, 0)):
        for j in range(size(uni, 1)):
            uni[i, j] = f(x[i], x[j])

    # positive, negative and net flows
    pos_flows = sum(uni, 1) / (uni.shape[0] - 1)
    neg_flows = sum(uni, 0) / (uni.shape[0] - 1)
    net_flows = pos_flows - neg_flows
    return net_flows

# PROMETHEE method: it calls the other functions
def promethee(x, p_fn, w):
    weighted_uni_net_flows = []
    total_net_flows = []
    for i in range(x.shape[1]):
        weighted_uni_net_flows.append(w[i] *
                                      uni_cal(x[:, i:i + 1], p_fn[i]))

    # print the weighted unicriterion preference
    # net flows
    for i in range(size(weighted_uni_net_flows, 1)):
        k = 0
        for j in range(size(weighted_uni_net_flows, 0)):
            k = k + round(weighted_uni_net_flows[j][i], 5)
        total_net_flows.append(k)

    return around(total_net_flows, decimals=4)

