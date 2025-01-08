from numpy import *
import math

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
    for i in range(size(weighted_uni_net_flows, 1)):
        k = 0
        for j in range(size(weighted_uni_net_flows, 0)):
            k = k + round(weighted_uni_net_flows[j][i], 5)
        total_net_flows.append(k)

    return around(total_net_flows, decimals=4)

