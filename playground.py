import normal_game as ng
import numpy as np
import time

start = time.time()

print('init game..')
G = ng.Game([2, 5])
u = [i for i in range(G.num_payoffs)]
print('init payoff..')
U = ng.Payoff(G, payoff_vector = u)

end = time.time()

print("The time of execution of above program is :",
	(end-start) * 10**3, "ms")








