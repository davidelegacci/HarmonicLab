# Plug in any number of players and any number of strategies per player

import normal_game as ng
import numpy as np


"""Manual metric generators = any two invertible matrices, one of dim = dim C0 and the other of dim = dim C1
They are used to generate the metric g0 = A * At, g1 = B * Bt
"""
# A = np.eye(6)
# B = np.eye(9)

# A[0][0] = 2
# B[2][1] = 3


"""Example of Eculidean harmonic payoff for 2x2 game"""
# u_eu = [2, 1, 1, 2, -1, 0, 1, 0]


"""Example of nonrtivial game harmonic wrt non-euclidean metric non diagonal, without fully mixed NE"""

# A = np.eye(4)
# B = np.eye(4)

# A[0][0] = 2
# B[2][1] = 3


# u_counterexample = [2, 7, 0, 0, 1, 0, 0, 1]

"""Example of game harmonic wrt euclidean metric with uniformly mixed NE, not harmonic wrt other metric"""
# A = np.eye(4)
# B = np.eye(4)

# A[0][0] = 2
# B[2][1] = 3
# u_eu = [2, 1, 1, 2, -1, 0, 1, 0]


"""Example of nullspace method failing to return all the basis elements of the harmonic matrix (gives 4 rather tnah 5)"""
# A = np.eye(4)
# B = np.eye(4)

# A[0][0] = 2
# A[0][1] = 3
# B[2][1] = 3

# u = [2, 7, 0, 0, 1, 0, 0, 1]

""" Example with diagonal metric and fully mixed NE """
# u_diagonal = [-3.5, 3.75, -1, 0, -1.75, -3, -2, 3]

####################################################################################################
# manual_metric_generators = [A,B]


"""Game. metric_type in ['euclidean', 'random', 'manual', 'diagonal'] """
G = ng.Game([3,3], metric_type = 'euclidean', manual_metric_generators = 0)

# u_random_harmonic = G.random_harmonic_payoff

# Payoff

# Good 2x2 example, Euclidean
# u = [43, 1, 3, 0, 5, 2, -1, 3]
# uN = [23., 0.5, 23., 0.5, 3.5, 3.5, 1., 1.]
# uP = [16., 4.5, -16., -4.5, 5.5, -5.5, -6., 6.]
# uH = [4., -4., -4, 4., -4., 4., 4., -4.]

# Good 2x3 example, Euclidean
# u = [1, 1, -1, 3, 4, 2, 6, 7, 8, 9, 1, 3]

# uN = [2., 2.500, 0.500, 2., 2.500, 0.500, 7., 7., 7., 4.333, 4.333, 4.333]
# uP = [-2.900, -0.400, -0.700, 2.900, 0.400, 0.700, 0.267, -0.733, 0.467, 3.400, -2.600, -0.800]
# uH = [1.900, -1.100, -0.800, -1.900, 1.100, 0.800, -1.267, 0.733, 0.533, 1.267, -0.733, -0.533]

# potential of uP = [-1.067, -2.067, -0.867, 4.733, -1.267, 0.533]

# RPS 3x3 Euclidean
# x = 1/2
# y = 1/3
# z = 1/4

# u_RPS = [0, -3*x, 3*y, 3*x, 0, -3*z, -3*y, 3*z, 0, 0, 3*x, -3*y, -3*x, 0, 3*z, 3*y, -3*z, 0]

# u_h = [-0.083, 0.708, 0.458, -1.083, 0.083, -0.708, -0.458, 1.083, 0.229, -0.854, -0.229, 0.854, -1.229, 1.854, 1.229, -1.854, -0.146, 0.146, -0.229, 0.229, 1.146, -1.146, -0.771, 0.771]

""" if payoff_vector == 0, generate random one; else plug in explicit one"""

# u_2x3_harmonic = [-4, 5, 0, -2, 1, 2, 3, 1, 3, -2, 0, -2]


# u_example_seminar_2x3 = [-3, 0, -3, 3, -3, 0, 3, -5, 3, 0, 0, 1]

# uN_example_seminar_2x3 = [0.0, -1.5, -1.5, 0, -1.5, -1.5, 1/3, 1/3, 1/3, 1/3, 1/3, 1/3]
# uP_example_seminar_2x3 = [-0.9, -1.5, -0.6, 0.9, 1.5, 0.6, 1.267, -3.333, 2.067, 1.067, -2.333, 1.267]
# uH_example_seminar_2x3 = [-2.1, 3, -0.9, 2.1, -3., 0.9, 1.4, -2, 0.6, -1.4, 2, -0.6]


# u_battle_sexes = [5, -1, -5, 1, 2, -2, -4, 4]

# u_prisoner_dilemma = [2, 0, 3, 1, 2, 3, 0, 1]

# uN_prisoner_dilemma = [2.500, 0.500, 2.500, 0.500, 2.500, 2.500, 0.500, 0.500]
# uP_prisoner_dilemma = [-0.500, -0.500, 0.500, 0.500, -0.500, 0.500, -0.500, 0.500]
# uH_prisoner_dilemma = [0.0, 0, 0.0, 0.0, 0, 0, 0, 0]

# u_matching_pennies = [1, -1, -1, 1, -1, 1, 1, -1]

# u_star_coin_2x4 = [-1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1]

# x = 1/3
# y = 1/3
# z = 1/3
# u = [0, -3*x, 3*y, 3*x, 0, -3*z, -3*y, 3*z, 0, 0, 3*x, -3*y, -3*x, 0, 3*z, 3*y, -3*z, 0]

# a = 1 
# b = 2
# c = 3
# d = 4
# e = 5
# f = 6
# u = np.array([ 0, a, b, c, -(a+b+c), -a, 0, d, e, a-d-e, -b, -d, 0, f, b+d-f, -c, -e, -f, 0, c+e+f, a+b+c, d+e-a, f-b-d, -(c+e+f), 0])

u = [ 1, 4, 5, 4, 2, 6, 5, 6, 3]

def make_zero_sum(u):
	v = - np.array(u)
	return list(u) + list(v)

print(make_zero_sum(u))


u_zero_sum = make_zero_sum(u)

U = ng.Payoff(G, payoff_vector = u_zero_sum)


print('\n--end--\n')

# G.draw_undirected_response_graph()
# U.draw_directed_response_graph()









