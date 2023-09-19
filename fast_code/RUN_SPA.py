# Plug in any number of players and any number of strategies per player

import normal_game as ng
import numpy as np
import single_price_auction as spa
import time
import utils

FIXED_VALUE = 1
NOW = time.time()
POT_FILE_PATH = f'potentialness/{NOW}_potentialness.txt'

############################################################

for value in np.linspace(0.8, 1, 5):
	SPA = spa.SinglePriceAuction(max_num_bids = 15, values = [FIXED_VALUE,value])
	G = ng.Game(SPA.effective_num_bids)
	U = ng.Payoff(game = G, payoff_vector = SPA.utility, value = value, pot_file_path = POT_FILE_PATH)

utils.plot_potentialness(POT_FILE_PATH, FIXED_VALUE)


# value = 1
# SPA = spa.SinglePriceAuction(max_num_bids = 3, values = [1,value])
# G = ng.Game(SPA.effective_num_bids, metric_type = 'euclidean', manual_metric_generators = 0)
# U = ng.Payoff(game = G, payoff_vector = SPA.utility, value = value)

print('\n--end--\n')