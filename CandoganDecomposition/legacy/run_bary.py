############################################################
from collections import Counter
import normal_game_minimal_euclidean as ng
import numpy as np
import time
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import yaml
import pandas as pd
#######################################
with open('config.yml', 'r') as file:
	config = yaml.safe_load(file)


results_dir = config['results_dir']
experiment_name = config['experiment_name']
results_dir += experiment_name
#######################################

start = time.time()
#######################################

def make_df(indices, data):

	df = pd.DataFrame({
	    'Row': [idx[0] for idx in indices],  # first value is the row
	    # 'Column': [f"{idx[1]}{idx[2]}" for idx in indices],  # combine second and third as the column
	    'Column': [''.join(map(str, idx[1:])) for idx in indices],  # Combine the remaining values for column index
	    'Value': data
	})

	# Pivot the table to get the desired format
	df_pivot = df.pivot(index='Row', columns='Column', values='Value')

	# Fill missing values with 0 (if any) and display the table
	# df_pivot = df_pivot.fillna(0)

	# Display the table
	return df_pivot




#######################################


SKELETONS = [  [5, 5, 5], [6, 6, 6]   ]

def bary_payoff(i, a,):

    m = max(a)

    if a[i] < m:
        return 0

    count_dict = Counter(a)

    return 60 / count_dict[a[i]]


for s in SKELETONS:
	players = range(len(s))
	G = ng.GameFull(s, **config)
	A = G.nodes
	u_packed = [  [bary_payoff(i,a) for a in A] for i in players  ]
	u_flat = [  uia 
				for ui in u_packed
				for uia in ui  ]

	print(f"\n{s}")
	print(A)
	print(u_flat)

	U = ng.PayoffFull(game = G, payoff_vector = u_flat, **config)

	u = u_flat[:G.num_strategy_profiles]

	up = U.round_matrix(U.uN + U.uP)[:G.num_strategy_profiles]

	uh = U.round_matrix(U.uH)[:G.num_strategy_profiles]

	payoffs_first_player = [u, up, uh]
	payoff_matrices = [make_df( A,p ) for p in payoffs_first_player]


	# Write the first DataFrame to the CSV file
	payoff_matrices[0].to_csv(f'bary_decomposition/bary-{s}.csv', index=True, sep = ";", decimal = ",")

	# # Append subsequent DataFrames with a blank line
	with open(f'bary_decomposition/bary-{s}.csv', 'a') as f:
	    for df in payoff_matrices[1:]:
	        f.write('\n')  # Write a blank line between DataFrames
	        df.to_csv(f , index=True,  sep = ";", decimal = ",")


	









# -----------------------------------------------------------
end = time.time()
print("\nEND\nThe time of execution of above program is :",
	(end-start) * 10**3, "ms")

