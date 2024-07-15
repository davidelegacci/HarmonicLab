############################################################
import normal_game	 as ng
############################################################
import numpy as np
import first_price_auction as fpa
import time
import utils
from tqdm import tqdm
import os
import yaml


with open('config.yml', 'r') as file:
	config = yaml.safe_load(file)

############################################################
# SINGLE AUCTION
############################################################
# start = time.time()
# SPA = fpa.SinglePriceAuction(n_discr = 5, values = [1,1])
# G = ng.Game(SPA.skeleton)
# U = ng.PayoffPotValue(game = G, payoff_vector = SPA.utility, **config)
# end = time.time()
# print("The time of execution of above program is :",
# 	(end-start) * 10**3, "ms")

############################################################

############################################################
# MULTIPLE AUCTIONS VARYING VALUE AND PLOTTING
############################################################


# Discretization
N_DISCR = 5

# Running value
LOW_VALUE = 0
HIGH_VALUE = 1
N_VALUES = 10

NOW = time.time()

results_dir = config['results_dir'] + '/' + config['experiment_name'] + f'/potentialness/{NOW}'
os.makedirs(results_dir)
potentialness_file  = results_dir + f'/pot.csv'
config['potentialness_file'] = potentialness_file

for running_value in tqdm(np.linspace(LOW_VALUE, HIGH_VALUE, N_VALUES)):
	SPA = fpa.SinglePriceAuction(n_discr = N_DISCR, values = [config['fixed_value'],running_value])
	G = ng.Game(SPA.skeleton)
	config['running_value'] = running_value
	U = ng.PayoffPotValue(game = G, payoff_vector = SPA.utility, **config)

utils.plot_value_potentialness_FPSB(potentialness_file, config['fixed_value'], N_DISCR, N_VALUES)

############################################################


print('\n--end--\n')