import matplotlib
import matplotlib.pyplot as plt
import numpy as np 
import math
import time


#load the dataset
data = numpy.loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

# add features
for i in range(6):
	for j in range(len(X)):
		X[j].append((X[j][0]*X[j][1])**i)

print X[1]