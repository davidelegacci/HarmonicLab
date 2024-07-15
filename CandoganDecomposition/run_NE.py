############################################################
import normal_game	 as ng
############################################################

import numpy as np
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
import yaml

start = time.time()


with open('config.yml', 'r') as file:
	config = yaml.safe_load(file)


import first_price_auction as fpa
SPA = fpa.SinglePriceAuction(n_discr = 3, values = [1,1])
G = ng.GameFull(SPA.skeleton, **config)
U_original = ng.PayoffNE(game = G, payoff_vector = SPA.utility)

# G = ng.GameFull([2,2], **config)
# u_original = np.random.randint(-5, 5, G.num_payoffs)
# U_original = ng.PayoffNE(game = G, payoff_vector = u_original, **config)

NOW = time.time()
results_dir = config['results_dir'] + '/' + config['experiment_name'] + f'/pureNE/{NOW}'
os.makedirs(results_dir)
i=0
for alpha in tqdm(np.linspace(0, 1, 3)):
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

print((end-start) * 10**3)