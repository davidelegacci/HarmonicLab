import sympy as sp
import numpy.random as rm

def make_mixed_strategy(n):
	# returns a list of n elements between 0 and 1 summing up to 1
	while True:
		a  = rm.random(n-1)
		p = 1 - sum(a)
		if p >= 0:
			break
	return list(a) + [p]


def factor(l, skeleton):
	'''
	Given [1, 2, 3, 4, 5] and [2, 3] returns [ [1,2], [3,4,5] ]
	'''
	assert len(l) == sum(skeleton)
	data = []
	i = 0
	j = skeleton[1]
	for n in skeleton:
		data.append( l[i:j] )
		i = j
		j +=n
	return data

def flatten(l):
	flattened_list = [item for sublist in l for item in sublist]
	return (flattened_list)


# switch between coords and points
def coords_points(L):

	def make(*values):
		return values

	return make(*zip(*L))


def format_u(u_string, players_names):
	u_string = u_string.replace('))', ')')

	# with space
	# u_string = u_string.replace(f'u({player}, ', f'\\, u^{player}')

	# without space
	for i in players_names:
		u_string = u_string.replace(f'u({i}, ', f'u^{i}')

	return u_string

def format_V(V_string, player):
	V_string = V_string.replace('))', ')')
	V_string = V_string.replace(f'u({player}, ', f'\\, u^{player}')
	V_string = V_string.replace ('\\right) + ', '\\right) \\\\ & + ')
	return V_string

def format_v(V_string, player):
	V_string = V_string.replace('))', ')')
	V_string = V_string.replace(f'u({player}, ', f'\\, u^{player}')
	return V_string



def solve_system(system, my_vars):

	'''
	Usage: in any expression containing my_vars make expression.subs(sol_dict)

	Note comma to unpack solution
	Sometimes this unpacking may fail if linsolve returns finiteset objects, in which case use .arg[0] to unpack solution
	'''

	sol, = sp.linsolve(system, my_vars)
	sol_dict = dict(zip(my_vars, sol))

	# Solution check
	for eq in system:
		assert eq.subs(sol_dict) == 0

	return sol_dict


def solve_poly_system(poly, my_vars):

	system = poly.coeffs()
	sol_dict = solve_system(system, my_vars)
	return sol_dict

def latex_print_dict(sol_dict, players_names, open_tex, close_tex):
	print(open_tex)
	print('\\begin{split}')
	for key in sol_dict:
	    latex_key = format_u( sp.latex(key), players_names ) 
	    latex_value = format_u( sp.latex( sol_dict[key] ), players_names )
	    print(f'{latex_key} &= {latex_value} \\\\ ')
	print('\\end{split}')
	print(close_tex)




# def print_dict(my_dict):
# 	[print(f'{e} : {my_dict[e]}') for e in my_dict]


