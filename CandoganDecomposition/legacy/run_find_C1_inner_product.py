import normal_game	 as ng
import yaml

#######################################
with open('config.yml', 'r') as file:
	config = yaml.safe_load(file)


G = ng.GameFull([2,3], **config)

# -------------------------------
# 2x2 generalized harmonic example of polaris seminar
# u = [1, -1, 0, 0, -1, 1, 0, -1]

u = [4, -6, 7, 3, -3, 2, 5, 8, 2, -4, -5, -3]

# 2x2 uniform harmonic
# u = [2, 3, 3, 2, -2, -3, -1, 0]
# -------------------------------
U = ng.PayoffFull(game = G, payoff_vector = u, **config)



# -----------------------------------------------------------
print("\nStarting computation to find metric such that my harmonic game is harmonic")

import sympy as sp
from pprint import pprint

# harmonic_matrix =  G.metric_0.flat_matrix @ G.boundary_1_matrix @ G.metric_1.sharp_matrix @ G.pwc_matrix 

# redefine sharp1

flat0 = sp.Matrix( G.metric_0.flat_matrix  )
bd1 = sp.Matrix( G.boundary_1_matrix  )
D = sp.Matrix(G.pwc_matrix)


sharp1 = sp.zeros( G.dim_C1 )


def make_symbols(N,s,shift = 0):
	my_symbols = []
	for i in range(shift, N + shift):
	    tmp_st = f'{s}{i}'
	    globals()[tmp_st] = sp.Symbol(tmp_st)
	    my_symbols.append(globals()[tmp_st])
	return my_symbols


sharp1_dofs = make_symbols( G.dim_C1, 's', shift = 1 )

for i in range(G.dim_C1):
	sharp1[i, i] = sharp1_dofs[i]

# ------------------------------------------------------
# If this is on, everything boils down to euclidean case
# sharp1 = sp.eye( G.dim_C1 )
# ------------------------------------------------------ù

# ------------------------------------------------------
# If this is on, equivalent to main code
# sharp1 = sp.Matrix(G.metric_1.sharp_matrix)
# ------------------------------------------------------

H =  flat0 * bd1 * sharp1 * D 

pprint(H)


u_sym = sp.Matrix(u)

Hu = H * u_sym

print("Sybolical check that current payoff is harmonic: Hu = ")
pprint(Hu)
print( f'Is harmonic? {all(element == 0 for element in Hu)}' )

eqs = [ row for row in Hu ]

sol = sp.linsolve(eqs, sharp1_dofs )


print("\neqs and sol")
print(eqs)
pprint(sol)

sol = sol.subs('s4', 1)



# sol_dict = dict(zip(sharp1_dofs, sol.args[0] ))

# H_sol = H.subs(sol_dict)

# pprint(H_sol * u_sym)

# -----------------------------------------------------------