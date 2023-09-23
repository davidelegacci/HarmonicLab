############################################################
import normal_game_FULL	 as ng
# import normal_game	 as ng
############################################################


import numpy as np
import time
import utils
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# start = time.time()

# G = ng.Game([5,5])
# u = np.random.randint(-5, 5, G.num_payoffs)

import first_price_auction as fpa
SPA = fpa.SinglePriceAuction(n_discr = 15, values = [1,1])
G = ng.Game(SPA.skeleton)
U_original = ng.Payoff(game = G, payoff_vector = SPA.utility)

# end = time.time()

# print("The time of execution of above program is :",
# 	(end-start) * 10**3, "ms")

NOW = time.time()
folder_path = f'pureNE/{NOW}'
os.makedirs(folder_path)
i=0
for alpha in tqdm(np.linspace(0, 1, 20)):
	u_alpha = utils.make_alpha_game(alpha, U_original.uP, U_original.uH)
	# print(f'\nALPHA = {alpha}')
	U_alpha = ng.Payoff(G, payoff_vector = u_alpha.reshape(G.num_payoffs))
	pure_NE, pure_NE_matrix = U_alpha.pure_NE
	ax = sns.heatmap(pure_NE_matrix, linewidth = 0.5)
	plt.title(alpha)
	plt.savefig(f'{folder_path}/{i}.pdf')
	plt.close()
	i+=1
