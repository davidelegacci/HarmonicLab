# Plug in any number of players and any number of strategies per player

import normal_game as ng
import numpy as np
import single_price_auction as spa

SPA = spa.SinglePriceAuction(max_num_bids = 3, values = [1,1])
SPA.print_game_size()
SPA.print_utility()
print(j)

"""Game. metric_type in ['euclidean', 'random', 'manual', 'diagonal'] """
G = ng.Game([3,3], metric_type = 'euclidean', manual_metric_generators = 0)

# first price 2 players auction, value of player one fixed to 1, value of player 2 is parameter, 3 strategies each

# general



# U = ng.Payoff(game = G, payoff_vector = u, value = v)

for v in np.linspace(0, 1, 50):
	# u = [v/2, 0, 0, v-0.5, v/2, 0, v-1, v-1, v/2, 0.5, 0.5, 0, 0, 0.5, 0, 0, 0, 0.5]
	u = [v/2, 0, 0, v-0.5, v/2 - 0.25, 0, v-1, v-1, (v-1)/2, 0.5, 0.5, 0, 0, 0.25, 0, 0, 0, 0]
	U = ng.Payoff(game = G, payoff_vector = u, value = v)

print('\n--end--\n')