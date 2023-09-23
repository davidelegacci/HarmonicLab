############################################################
import normal_game_FULL	 as ng
# import normal_game	 as ng
############################################################
import numpy as np
import first_price_auction as fpa
import time
import utils
from tqdm import tqdm


############################################################
# SINGLE AUCTION
############################################################
SPA = fpa.SinglePriceAuction(n_discr = 3, values = [1,1])
G = ng.Game(SPA.skeleton)
U = ng.Payoff(game = G, payoff_vector = SPA.utility)
############################################################

############################################################
# MULTIPLE AUCTIONS VARYING VALUE AND PLOTTING
############################################################

# FIXED_VALUE = 0.5
# N_DISCR = 5

# LOW_VALUE = 0
# HIGH_VALUE = 1
# N_VALUES = 5

# NOW = time.time()
# POT_FILE_PATH = f'potentialness/{NOW}_potentialness.csv'

# for value in tqdm(np.linspace(LOW_VALUE, HIGH_VALUE, N_VALUES)):
# 	SPA = fpa.SinglePriceAuction(n_discr = N_DISCR, values = [FIXED_VALUE,value])
# 	G = ng.Game(SPA.skeleton)
# 	U = ng.Payoff(game = G, payoff_vector = SPA.utility, plot_auction_potentialness = True, value = value, pot_file_path = POT_FILE_PATH)

# utils.plot_value_potentialness_FPSB(POT_FILE_PATH, FIXED_VALUE, N_DISCR, N_VALUES)

############################################################


print('\n--end--\n')