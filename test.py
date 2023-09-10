import sympy as sp
import numpy as np




A = [
[ 1,  0, -1,  0,  1, -1,  0,  0],
[ 0,  1,  0, -1, -1,  1,  0,  0],
[-1,  0,  1,  0,  0,  0,  1, -1],
[ 0, -1,  0,  1,  0,  0, -1,  1]]

A = sp.Matrix(A)

var = []
for i in range(1, A.shape[1]+1):
    tmp_st = f'x{i}'
    globals()[tmp_st] = sp.Symbol(tmp_st)
    var.append(globals()[tmp_st])

print(var)



zero = sp.Matrix(np.zeros(A.shape[0]))

system = A, zero

S = sp.linsolve(system, var)

S = S.args[0]

print(S)

print(S)