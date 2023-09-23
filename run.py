############################################################
# import normal_game_FULL	 as ng
import normal_game	 as ng
############################################################


import numpy as np
import time

start = time.time()

G = ng.Game([2,2])
# u = range(G.num_payoffs)
u = [1, 3, 2, 8, 4, 2, 3, 1]
U = ng.Payoff(G, payoff_vector = u)

end = time.time()

print("The time of execution of above program is :",
	(end-start) * 10**3, "ms")