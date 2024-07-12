# Make interpolation alpha game between a potential and a harmonic one

import numpy as np

'''
u normalized

u = [2.0, 3.5, -2.0, -3.5, 1.5, -1.5, 5.5, -5.5]
--> uP = [1.375, 4.125, -1.375, -4.125, 2.125, -2.125, 4.875, -4.875]
--> uH = [0.625, -0.625, -0.625, 0.625, -0.625, 0.625, 0.625, -0.625]

Potentialness = 0.8460461415885743
Potentialness new = 0.9838438483563934

Converges to pure NE
'''

'''
Divide by all by 2; potentialness is invariant; components scale well; dynamics and NE are the same
--> u = [1.0, 1.75, -1.0, -1.75, 0.75, -0.75, 2.75, -2.75]
uP = [0.6875, 2.0625, -0.6875, -2.0625, 1.0625, -1.0625, 2.4375, -2.4375]
uH = [0.3125, -0.3125, -0.3125, 0.3125, -0.3125, 0.3125, 0.3125, -0.3125]

Potentialness = 0.8460461415885743
Potentialness new = 0.9838438483563934

Converges to pure NE
'''

'''
Break convergence increasing harmonicity
Example of 2x2 game with pure NE but non-convergence
Obtained with alpha = 0.2

u = [0.775, 0.325, -0.775, -0.325, -0.075, 0.075, 1.475, -1.475]
Du = [0.15, -1.55, -0.65, -2.95]
DuP = [-0.85, -0.55, -1.65, -1.95]
DuH = [1.0, -1.0, 1.0, -1.0]
Potentialness = 0.5787457279961593
Potentialness new = 0.8085045786386456

'''
up = [1.375, 4.125, -1.375, -4.125, 2.125, -2.125, 4.875, -4.875]
uh = [0.625, -0.625, -0.625, 0.625, -0.625, 0.625, 0.625, -0.625]

up = np.array(up)
uh = np.array(uh)

u = up + uh

def make_ua(a):
	return a * up + (1 - a) * uh

def show(iter):
	print(list(iter))


ua = make_ua(0.2)

show(ua)
