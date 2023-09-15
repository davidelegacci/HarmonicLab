import numpy as np
import matplotlib.pyplot as plt



def two_tuples_differ_for_one_element(tuple_one, tuple_two):
    L = len(tuple_one)
    assert L == len(tuple_two)
    ary_one, ary_two = np.array(tuple_one), np.array(tuple_two)

    list_diff = list(ary_one - ary_two)
    number_of_zeros = list_diff.count(0)

    return True if number_of_zeros == L-1 else False 

def coords_points(L):
    def make(*values):
        return values

    return make(*zip(*L))


def are_same_edge(edge1, edge2):
    # edge = list of two points
    p1, p2 = edge1
    if p1 in edge2 and p2 in edge2:
        return True
    else:
        return False

def different_index(edge):
    a, b = edge
    i = 0
    while a[i] == b[i]:
        i+=1
    return i

def is_zero(ary, tolerance = 1e-6):
    # return not np.any(ary)
    for el in ary:
        if el > tolerance:
            return False
    return True

def orange(text):
    COLOR = '\033[93m'
    END = '\033[0m'
    return f'{COLOR}{text}{END}'

def red(text):
    COLOR = '\033[91m'
    END = '\033[0m'
    return f'{COLOR}{text}{END}'
