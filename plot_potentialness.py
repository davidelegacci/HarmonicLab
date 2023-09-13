import numpy as np
import matplotlib.pyplot as plt

with open('potentialness.txt') as f:
    data = f.readlines()

points = [p[1:-2].split(", ") for p in data]
points = [(float(p[0]), float(p[1]))for p in points]


def make_plot():
	plt.plot(*zip(*points), 'ro', ms = 3)
	plt.xlim(0,1)
	plt.ylim(0,1)
	plt.ylabel('Value')
	plt.xlabel('potentialness')
	plt.show()

make_plot()
