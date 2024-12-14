

# Candogan

############################################################
import normal_game_minimal_euclidean as ng


# import normal_game_full as ng



############################################################

# quick list for size of u
# 2      AN = 2
# 2x1    AN = 4
# 2x2: 	 AN = 8
# 2x3: 	 AN = 12
# 3x3: 	 AN = 18
# 2x2x2: AN = 24
# 4x4: 	 AN = 32

import numpy as np
import time
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import yaml

#######################################
with open('config.yml', 'r') as file:
	config = yaml.safe_load(file)


results_dir = config['results_dir']
experiment_name = config['experiment_name']
results_dir += experiment_name
#######################################

start = time.time()

print('\n\n\n\n\n\n')

# ---------------------------------
G = ng.GameFull([2, 2, 2], **config)
# ---------------------------------

# u = [1, -2, -2, -1, 2, -3, -3, -1]
# u = [3, 1, 2, 0, 0, 1, 0, 0] + [1, 3, 0, 2, 1, 0, 0, 0] +[3, 3, 2, 2, 1, 1, 0, 0]

# u = [-1, 2, 3, 6, +1, -2, -3, -6]

u = np.random.randint(-5, 5, G.num_payoffs) 
# u = np.arange(1, G.num_payoffs + 1)
# ---------------------------------

# u = [20.0, 0, 0, 0, 0, 0, 0, 0, 0, 60.0, 30.0, 0, 30.0, 20.0, 0, 0, 0, 0, 60.0, 60.0, 30.0, 60.0, 60.0, 30.0, 30.0, 30.0, 20.0, 20.0, 0, 0, 60.0, 30.0, 0, 60.0, 60.0, 30.0, 0, 0, 0, 30.0, 20.0, 0, 60.0, 60.0, 30.0, 0, 0, 0, 0, 0, 0, 30.0, 30.0, 20.0, 20.0, 60.0, 60.0, 0, 30.0, 60.0, 0, 0, 30.0, 0, 30.0, 60.0, 0, 20.0, 60.0, 0, 0, 30.0, 0, 0, 30.0, 0, 0, 30.0, 0, 0, 20.0]


# u = [0.9, -0.6, -0.3, -0.9, 0.6, 0.3, -0.6, 0.4, 0.2, 0.6, -0.4, -0.2]

# -------------------
# Linear combination of harmonic generated by candogan code, 2x3, measure = [  [1, 2], [1, 2, 1]  ]
# basis = [
# [1.00000000000000, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 1.00000000000000, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 1.00000000000000, 0, 0, 1, 0, 0, 0, 0, 0, 0],
# [-2.00000000000000, 2.00000000000000, -2.00000000000000, 0, 0, 0, 0, -2.00000000000000, 0, 0, 1, 0]
# ] 

# basis = [ np.array(e) for e in basis ]


# N = len(basis)
# A = np.random.randint(-5, 5, N)
# u = sum([ A[i] * basis[i] for i in range(N) ])

# -------------------



# u = [-5, 2, 0, -3, 0, 2, 4, 2, 4, -2, -1, -2]
# u = [0, 5, -6, -1, 4, -3, 0, 0, 2, -3, -3, -4]

# u = [4.50000000000000, 2.00000000000000, -5.50000000000000, 2, 2, -3, -3.00000000000000, -2.50000000000000, -2, 1, -1, -3]

# u = [1.50000000000000, -4.00000000000000, -1.25000000000000, -1, 1, 0, 0.250000000000000, 1.25000000000000, 1, 2, -2, -1]

# u = [0.267949192431123, 10.6602540378444, -9.92820323027551, 2, 2, -3, -6.00000000000000, -12.0000000000000, -3, -2, 0, -3]


# 2x3 gen harm
# u = [-5, 2, 0, -3, 0, 2, 4, 2, 4, -2, -1, -2]
# u = [4, -6, 7, 3, -3, 2, 5, 8, 2, -4, -5, -3]

# u = [19.7261622014082, -6.89897948556636, -6.92820323027551, 3, -2, 0, -12.0000000000000, 0, 0, 1, -3, -3]

# 2x2x2
# u = [-2.00000000000000, 3.00000000000000, -2.00000000000000, 2.00000000000000, -1, 3, 0, -1, 0, -1.00000000000000, 0, -2, 1, 3, 2, 3, -2.50000000000000, -3, -1, -3, 0, 0, 0, 3]

# 3x3

# u = [3.33333333333333, -5.66666666666667, -4.66666666666667, 0.666666666666667, -3.33333333333333, -4.33333333333333, -1, -3, -3, -2.00000000000000, 4.00000000000000, 2, -2.00000000000000, -3, -2, 2, -1, -1]

# u = [2.33333333333333, -7.66666666666667, 1.33333333333333, 1.66666666666667, -2.33333333333333, -3.33333333333333, 1, -3, -2, 2.00000000000000, 6.00000000000000, 0, 0, -2, 2, 2, 0, 2]

# 2x3 unif harm
# u = [1.00000000000000, 2.00000000000000, 5.00000000000000, 2, 3, 3, 0, 0, -1, 0, 0, 1]

# 2x3 generalized harmonic
# u = [8.00000000000000, -4.00000000000000, 1.00000000000000, 3, 0, 2, -2.00000000000000, 1.00000000000000, 0, 3, 0, 1]

# -------------------------------
# 2x2 generalized harmonic example of polaris seminar
# u = [1, -1, 0, 0, -1, 1, 0, -1]

# 2x2 uniform harmonic
# u = [2, 3, 3, 2, -2, -3, -1, 0]
# -------------------------------

# u = [1.00000000000000, -2.00000000000000, -1, 0, -3.00000000000000, -1, 1, -1]
# u = [0, -3.00000000000000, -1, -2, -1.00000000000000, 0, 3, 2]

# u = [3.00000000000000, 1.00000000000000, 9.00000000000000, -9.00000000000000, 1, 2, 3, -2, 7.00000000000000, -8.00000000000000, 1, -1, -1, 2, 1, -1, -9.00000000000000, -1, -3, -3, 2, -2, 1, -3]

# u = [-5.00000000000000, 0, 4.00000000000000, -1, 1, -1, 5.00000000000000, 4.00000000000000, 2, -2, -1, 1]
# u = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
#u = [7, -9, 3, -7, 9, 1, -7, -3, -1, 5, 0, 1, 3, 2, 3, 0, -3, 2, -3, -1, -1, -2, 3, 0]
#u = [1, 2, 3, 4, 5, 6, 7, 8]

# 4x4
# u = [0.0, 1.0, -2.0, 1.0, -1.0, 0.0, 3.5, -2.5, 2.0, -3.5, 0.0, 1.5, -1.0, 2.5, -1.5, 0.0, 0.0, -1.0, 2.0, -1.0, 1.0, 0.0, -3.5, 2.5, -2.0, 3.5, 0.0, -1.5, 1.0, -2.5, 1.5, 0.0]
# u = [0.0, 0.3310862697506767, 0.9741583642123127, -1.3052446339629893, -0.3310862697506767, 0.0, 0.17121535296633528, 0.1598709167843414, -0.9741583642123127, -0.17121535296633528, 0.0, 1.145373717178648, 1.3052446339629893, -0.1598709167843414, -1.145373717178648, 0.0, 0.0, -0.3310862697506767, -0.9741583642123127, 1.3052446339629893, 0.3310862697506767, 0.0, -0.17121535296633528, -0.1598709167843414, 0.9741583642123127, 0.17121535296633528, 0.0, -1.145373717178648, -1.3052446339629893, 0.1598709167843414, 1.145373717178648, 0.0]

# 5x5
# u = [0.0, 0.07668359805081815, 0.020267545581826307, 0.5352050823224256, -0.63215622595507, -0.07668359805081815, 0.0, 0.12322132825896137, 0.45828090250203735, -0.5048186327101806, -0.020267545581826307, -0.12322132825896137, 0.0, 0.7751018777858568, -0.6316130039450691, -0.5352050823224256, -0.45828090250203735, -0.7751018777858568, 0.0, 1.7685878626103197, 0.63215622595507, 0.5048186327101806, 0.6316130039450691, -1.7685878626103197, 0.0, 0.0, -0.07668359805081815, -0.020267545581826307, -0.5352050823224256, 0.63215622595507, 0.07668359805081815, 0.0, -0.12322132825896137, -0.45828090250203735, 0.5048186327101806, 0.020267545581826307, 0.12322132825896137, 0.0, -0.7751018777858568, 0.6316130039450691, 0.5352050823224256, 0.45828090250203735, 0.7751018777858568, 0.0, -1.7685878626103197, -0.63215622595507, -0.5048186327101806, -0.6316130039450691, 1.7685878626103197, 0.0]
# u = [0, 1, 1, -2, -1, 0, -2, 3, 1, 2, 0, -1, 0, -3, 1, 0, 0, -1, 1, 0, 1, 0, 2, -3, 1, -2, 0, 1, -2, 3, -1, 0]
# u = [5, -1, 1, 7, 1, -3, 3, 3, -1, 5, 7, 3, -1, 1, 3, 1, -3, -1]
# u = [1, 2, -3, 4, 5, -9, -5, -7, 12]
# u += list(-np.array(u))

# u = [5, -9, -7, 3, 6, 11, -3, -7, -1, -2, 8, 0, -4, -2, -4, 1, -3, -3, -1, 4, 0, 0, 1, -4, 1, 5, 11, 8, 1, 0, -9, -3, 0, -3, 0, -7, -7, -4, -3, 1, 3, -1, -2, -1, -4, 6, -2, -4, 4, 1]

# u = [1, 2, -1, 3, 1, 4, 6, 2, -3, 1, 3, 6, 2, 1, 2, -1, 4, -3]
# u = [1, -6, -8, 7, 8, -5, -5, -9, -1, 1, 7, -5, -6, 8, -9, -8, -5, -1]

# u = np.random.randint(-5, 5, 18 )

# u = [4.00000000000000, 5.00000000000000, -6.00000000000000, 3.00000000000000, 3, 3, -1, 1, -6.00000000000000, -3.00000000000000, -3, -3, 3, -1, 1, -2, -1.00000000000000, -3, 0, -2, -2, -1, -3, 0]
# u = [2.00000000000000, -3.00000000000000, 4.00000000000000, 2.00000000000000, 2, -3, 3, 3, 5.00000000000000, -3.00000000000000, 2, 0, 3, -1, 0, 2, -2.00000000000000, 1, 2, 0, -2, 1, 1, -3]

# 2x2 potential
# u = [0, -1, 3, 0, 0, 1, -1, -2]

# 2x2 harmonic (pure + non-strategic, 5 dofs)
# [a, b, c, d, e] = np.random.randint(-10, 10, 5)
# u_h = np.array([ a, -a, -a, a, -a, a, a, -a  ])
# u_n = np.array([ c, d, c, d, e, e, b, b ])

# u = u_h  + u_n 	# harmonice + non-strategic
# end 2x2

# 2x2 coexact
# u = [-6, -1, 2, -9, 0, -8, -5, 3]

# 2x2 exact
# u = [21, -5, 9, -3, 5, -7, 7, 9]

# 2x2 random + NS
# u = np.array([0, 1, 2, -1, 3, -1, 2, 0]) + np.array([2, 3, 2, 3, 4, 4, 5, 5])

# 2x2 NS
# u = [2, 3, 2, 3, 4, 4, 5, 5]

# u = [0.775, 0.325, -0.775, -0.325, -0.075, 0.075, 1.475, -1.475]
# u = np.random.randint(-10, 10, 8)

##############################################################
# random
# u = np.random.randint(-10, 10, 81)
# u = [2, 0, 3, 1, 2, 3, 0, 1]  


# 2x3 exact --> potential OK
# u = [-14, -4, -5, 3, -3, -1, -9, 2, 1, 5, 0, 2]

# 2x3 coexact --> harmonic OK
# u = [11, -10, -2, 3, -3, -1, -2, 3, 1, 5, 0, 2]

# 2x3 potential --> exact OK
# u = [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6]

# 2x3 harmonic --> coexact OK
# u = [-7, 13, -9, -3, 2, -2, 2, -3, 3, -2, 3, -3]
##############################################################

# 2x3 harmonic --> coexact OK

# a = 3
# b = 0

# u1 = np.array( [ a, b, -(a+b), -a, -b, a+b ] )
# u2 = - 2/3 * u1

# u = np.concatenate( (u1, u2) ).tolist()

# u = [2, 0, 3, 1, 4, 6, 0, 2]

## 2x2x2
# harmonic
# u = [-21, -16, 2, 21, -4, -8, -9, 7, 6, 24, 9, -4, -7, -8, 3, 7, 20, 0, -7, 7, 2, 9, -2, -3]

# potential
# u = [-7, -10, -39, -8, -4, -8, -9, 7, 13, -6, 9, -4, -20, -8, 3, 7, -8, 0, -7, 7, 2, 9, -2, -3]

##### 2x2
# harm
# u = [-6, -1, 2, -9, 0, -8, -5, 3]

# generalized harmonic 2x2
# u_gen_harm = np.array([-1, -4, -3, -1, 1, 2, 2, -2])
# u_ns = np.array([-2.0, -2.5, -2.0, -2.5, 1.5, 1.5, 0.0, -0.0])
# u = u_gen_harm - u_ns

U = ng.PayoffFull(game = G, payoff_vector = u, **config)  
# U = ng.Payoff(game = G, payoff_vector = u, **config)

end = time.time()


# # -----------------------------------------------------------
# # Start computation to solve system to find harmonic measure, knowing is harmonic

# print("\n----")

# edges = G.networkx_graph.edges
# skeleton = G.num_strategies_for_player
# players = range(G.num_players)

# # mu_values = [ np.random.randint( 1, 5, skeleton[i]) for i in players  ]

# mu_values = [  [1, 3], [1, 2, 1]  ]

# # print(mu_values)

# pures_play = [ p.strategies for p in G.players ]


# # list of dicts
# mu = [ dict(zip(pures_play[i], mu_values[i])) for i in players  ]
# pures = G.nodes
# # print(pures)
# # print(mu)


# # P stands for product; this is product measure
# muPvalues = [      np.prod(  [ mu[i][  a[i]  ]  for i in players   ]   )       for a in pures    ]
# # print(muPvalues)
# muP = dict(zip(pures, muPvalues))


# for a in pures:
# 	print( a, muP[a] )

# # for e in edges:
# # 	print(e)


# # measure on C1 space
# muEvalues = [    muP[ e[0] ] *  muP[ e[1] ]     for e in edges    ]
# # print(muEvalues)

# muE = dict(zip(edges, muEvalues))

# for e in muE:
# 	print(f"{e}: {muE[e]}")





# -----------------------------------------------------------

print("\nEND\nThe time of execution of above program is :",
	(end-start) * 10**3, "ms")

