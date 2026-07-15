from sympy import Matrix, zeros
from sympy.solvers.solveset import linsolve
import numpy as np
import pprint
import sympy as sp

"""
Solve symbolycally linear system given by degenerate matrix
"""

def find_kernel(A, variables):

	"""A is numpy matrix. Outputs kernel of A in parametric form and explicit basis of kernel of A using Sympy
	"""


	# Sympy matrix
	A = Matrix(A)

	# test_variables = []
	# for i in range(1, A.shape[1]+1):
	#     tmp_st = f'x{i}'
	#     globals()[tmp_st] = sp.Symbol(tmp_st)
	#     test_variables.append(globals()[tmp_st])

	# Vector of zeros with dimension = number of rows of A
	zero = Matrix(np.zeros(A.shape[0]))

	# System Ax=0
	system = A, zero

	# Parametric form of kernel
	S = linsolve(system, variables)

	# Basis of kernel
	B = A.nullspace()

	# pprint.pprint(A)
	# print('\nParametric kernel\n')
	# [print(s) for s in S]
	# print('\nBasis of kernel\n')
	# print(B)
	return S, B


def my_linsolve(A,b):
	"""
	A is numpy matrix
	b is numpy array
	"""

	A = Matrix(A)
	b = Matrix(b)

	# non-homogeneous system Ax=b
	system = A, b

	S = linsolve(system)

	[print(s) for s in S]
	return S

# def test_linsolve():
# 	A = [
# 		[1, 1, 1],
# 		[0, 0, 0]
# 		]


# 	b = [0,0]

# 	my_linsolve(A,b)


# def test_kernel():
# 		A = [
# 			[1, 1, 1],
# 			[1, 0, 0]
# 			]
# 		print(find_kernel(A))

# test_kernel()


