############################################################
import normal_game	 as ng
############################################################

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
experiment_name = '/test'
results_dir += experiment_name
#######################################

start = time.time()

G = ng.GameFull([2,2], **config)
u = np.random.randint(-5, 5, G.num_payoffs)
U = ng.PayoffFull(game = G, payoff_vector = u, **config)

end = time.time()

print("The time of execution of above program is :",
	(end-start) * 10**3, "ms")

