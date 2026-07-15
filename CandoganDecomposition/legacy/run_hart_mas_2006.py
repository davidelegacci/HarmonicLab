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
import seaborn as sns

#######################################
with open('config.yml', 'r') as file:
	config = yaml.safe_load(file)

#######################################

start = time.time()

G = ng.GameFull([3,3], **config)
u_original = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1]
U_original = ng.PayoffFull(game = G, payoff_vector = u_original, **config)


# Make alpha games
NOW = time.time()
results_dir = config['results_dir'] + '/' + config['experiment_name'] + f'/pureNE/{NOW}'
os.makedirs(results_dir)
i=0
for alpha in tqdm(np.linspace(0, 1, 30)):
	print(utils.orange(f'\n--- alpha = {alpha} ------'))
	u_alpha = utils.make_alpha_game(alpha, U_original.uP, U_original.uH)
	U_alpha = ng.PayoffNE(G, payoff_vector = u_alpha.reshape(G.num_payoffs))
	pure_NE, pure_NE_matrix = U_alpha.pure_NE
	ax = sns.heatmap(pure_NE_matrix, linewidth = 0.5)
	ax.invert_yaxis()
	plt.title(alpha)
	plt.savefig(f'{results_dir}/{i}.pdf')
	plt.close()
	i+=1

end = time.time()

print("The time of execution of above program is :",
	(end-start) * 10**3, "ms")

