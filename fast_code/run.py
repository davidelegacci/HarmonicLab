import normal_game as ng
import numpy as np
import time

start = time.time()

G = ng.Game([2, 5, 3])
u = [i for i in range(G.num_payoffs)]
U = ng.Payoff(G, payoff_vector = u)

end = time.time()

print("The time of execution of above program is :",
	(end-start) * 10**3, "ms")