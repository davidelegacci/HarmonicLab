# Plug in any number of players and any number of strategies per player

import normal_game_FULL as ng
import numpy as np
import single_price_auction as spa

# SPA = spa.SinglePriceAuction(max_num_bids = 3, values = [1,1])
# SPA.print_game_size()
# SPA.print_utility()

"""Game. metric_type in ['euclidean', 'random', 'manual', 'diagonal'] """
# G = ng.Game([3,3], metric_type = 'euclidean', manual_metric_generators = 0)

# first price 2 players auction, value of player one fixed to 1, value of player 2 is parameter, 3 strategies each

# general payoff

# U = ng.Payoff(game = G, payoff_vector = u, value = v)


FIXED_VALUE = 0.5

for value in np.linspace(0, 1, 10):
	SPA = spa.SinglePriceAuction(max_num_bids = 5, values = [FIXED_VALUE,value])
	G = ng.Game(SPA.effective_num_bids, metric_type = 'euclidean', manual_metric_generators = 0)
	U = ng.Payoff(game = G, payoff_vector = SPA.utility, value = value)


# value = 1
# SPA = spa.SinglePriceAuction(max_num_bids = 3, values = [1,value])
# G = ng.Game(SPA.effective_num_bids, metric_type = 'euclidean', manual_metric_generators = 0)
# U = ng.Payoff(game = G, payoff_vector = SPA.utility, value = value)

print('\n--end--\n')