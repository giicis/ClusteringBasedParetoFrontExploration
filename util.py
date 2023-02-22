import math

import numpy as np


def boolean_distance(str1: str, str2: str) -> int:
    """
    Return distance between two lists of boolean.
    :param str1: First list of boolean
    :param str2: Second list of boolean
    :return: number of elements in common between the two input strings
    """
    r1_boolean = np.array(list(str1), dtype=int)
    r2_boolean = np.array(list(str2), dtype=int)
    return sum(r1_boolean != r2_boolean)


def jaccard_index(str1: str, str2: str) -> float:
    """
    Returns Jaccard Index between two lists of boolean
    :param str1: First list of boolean
    :param str2: Second list of boolean
    :return:
    """
    r1_boolean = np.array(list(str1), dtype=int)
    r2_boolean = np.array(list(str2), dtype=int)
    intersection = np.fmin(r1_boolean, r2_boolean)
    union = np.fmax(r1_boolean, r2_boolean)
    sum_union = sum(union)
    if sum_union == 0:
        return 0.
    else:
        return sum(intersection) / sum_union


def jaccard_distance(str1: str, str2: str) -> float:
    """
    Returns Jaccard Distance between two lists of boolean
    :param str1: First list of boolean
    :param str2: Second list of boolean
    :return:
    """
    return 1. - jaccard_index(str1, str2)


def distance(sol1: list, sol2: list, max_profit, max_cost) -> float:
    """
    Distance between two quasi-optimal solutions
    :param sol1: first solution
    :param sol2: second solution
    :return: distance
    """
    c1 = sol1[1] / max_profit
    c2 = sol2[1] / max_profit
    p1 = sol1[2] / max_cost
    p2 = sol2[2] / max_cost
    euc_dist = math.sqrt((c1 - c2) ** 2 + (p1 - p2) ** 2)
    # d_req = boolean_distance(sol1[3], sol2[3]) / len(sol1[3])
    d_req = jaccard_distance(sol1[3], sol2[3])
    # d_stk = boolean_distance(sol1[4], sol2[4]) / len(sol1[4])
    d_stk = jaccard_distance(sol1[4], sol2[4])
    return 1 * euc_dist + 1 * d_req + 1 * d_stk
