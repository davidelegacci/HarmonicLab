import numpy as np
import matplotlib.pyplot as plt
import csv

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


# def plot_potentialness_txt(file_path, fixed_value, n_discr, n_values):
#     with open(file_path) as f:
#         data = f.readlines()

#     cleaned_data = [p[1:-2].split(", ") for p in data]
#     points = [(float(p[0]), float(p[1]))for p in cleaned_data]

#     plt.plot(*zip(*points), 'ro')
#     plt.xlim(0,1)
#     plt.ylim(0,1)
#     plt.ylabel('Value')
#     plt.xlabel('potentialness')
#     plt.title(f'Fixed value = {fixed_value}, n_discr_bids = {n_discr}, n_values = {n_values}')
#     plt.show()

def plot_value_potentialness_FPSB(file_path, fixed_value, n_discr, n_values):

    with open(file_path) as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        data = list(reader)
        print(data)

    # now data is a list of "points", each string

    points = [(float(p[0]), float(p[1]))for p in data]
    x,y = coords_points(points)
    skeletons = [p[2] for p in data]

    fig, ax = plt.subplots()
   
    scatter = ax.scatter(x,y)


    # equivalent to skel = skeletons[i] for i in range(len(skeletons))
    for i, skel in enumerate(skeletons):
        if i == 0 or (i > 0 and skel != skeletons[i-1]):
            ax.annotate(skel, (x[i], y[i]))
        
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.ylabel('Value')
    plt.xlabel('potentialness')
    plt.title(f'Fixed value = {fixed_value}, n_discr_bids = {n_discr}, n_values = {n_values}')
    plt.show()


def make_alpha_game(alpha, uP, uH):
    return alpha * np.array(uP) + (1 - alpha) * np.array(uH)










