import numpy as np
import numpy.linalg as la
from metric import *
import sympy as sp

"""Playground to test pinv"""

g = Metric(RandomMetric(2).matrix)

G = Metric(RandomMetric(2).matrix)

A = np.array([
	[2, 4], 
	[1, 2],
	])
print('A')
print(sp.latex(sp.Matrix(A)))
print()
print('determinant')
print(la.det(A))
print()

A = Homomorphism(A, g, G)
print('pinv')
print(sp.latex(sp.Matrix(A.pinv)))

print('g')
print(sp.latex(sp.Matrix(g.matrix)))
print('G')
print(sp.latex(sp.Matrix(G.matrix)))


#######################
# Basic test
# print('A: V --> W')
# A = np.array([
# 	[1, 0, 1], 
# 	[1, 0, 0],
# 	[0, 0, 1]
# 	])

# print(A)

# try:
# 	Ai = la.inv(A)
# 	print('\nInverse of A')
# 	print(Ai)
# except:
# 	print('\nA not invertible')

# print('\nB : W --> V = pinv of A')
# B = la.pinv(A)
# print(B)


# print('\nAB: W --> W')
# print(A @ B)

# print('\nBA: V --> V')
# print(B@A)

# print('\nABA')
# print(A@B@A)

# print('\nBAB')
# print(B@A@B)

# print('\nABAB')
# print(A@B@A@B)

# print('\nBABA')
# print(B@A@B@A)
#######################