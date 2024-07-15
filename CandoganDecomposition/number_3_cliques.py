from scipy.special import binom as b
from numpy import prod
import numpy as np


Ai = [3,3,3]
A = prod(Ai)

num_edges_in_i_cluster = [1/2 * ai * (ai-1) for ai in Ai]
num_nodes_in_i_cluster = Ai
num_i_clusters = [A / ai for ai in Ai]

num_i_edges = 1/2 * A * np.array([ai-1 for ai in Ai])
num_edges = sum(num_i_edges)

num_i_edges_per_node = [ai-1 for ai in Ai]
num_edges_per_node = sum(num_i_edges_per_node)

num_3_cliques_in_i_cluster = [b(ai, 3) for ai in Ai]
num_3_cliques_per_player_i = [A/6 * (ai-1) * (ai-2) for ai in Ai]
num_3_cliques = sum(num_3_cliques_per_player_i)


data = {'game': Ai,
		'num_edges':num_edges,
		'num_nodes':A,

		'num_i_clusters':num_i_clusters,
		'num_edges_in_i_cluster':num_edges_in_i_cluster,
		'num_nodes_in_i_cluster':num_nodes_in_i_cluster,
		
		'num_i_edges':num_i_edges,
		'num_i_edges_per_node':num_i_edges_per_node,
		'num_edges_per_node':num_edges_per_node,

		'num_3_cliques_in_i_cluster':num_3_cliques_in_i_cluster,
		'num_3_cliques_per_player_i':num_3_cliques_per_player_i,
		'num_3_cliques':num_3_cliques
		}

[print(k, ':', data[k]) for k in data]

