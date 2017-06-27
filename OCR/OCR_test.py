# Based on ML-Coursera-ex2, uses downhill simplex algorithm
from __future__ import division
import os
import numpy as np 
from processData import mergeFiles
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):
	return 1 / (1 + np.exp(-z + 1e-24))

def cost_function(W, X, Y, epsilon=0.1):
	reg = epsilon * (W * W).mean()
	J =  -np.dot(Y,np.log(sigmoid(X.dot(W.T)))) - np.dot((1-Y),np.log(1-sigmoid(X.dot(W.T))))
	return J + reg

def grad(X, Y, W, epsilon=0.1):
	Y = Y.reshape(len(Y),1)
	reg = epsilon * W
	grad = np.dot((sigmoid(X.dot(W.T)) - Y).T, X)
	return (grad + reg) / len(X)

def train_gradient_descent(X, Y, alpha = 0.000001, iterations = 10000):
	# Initialize weights
	costs = []
	W = np.zeros((1,X.shape[1]))
	print "Runnning"
	for i in range(iterations):
		costs.append(float(cost_function(W, X, Y)))
		update = alpha * grad(X, Y, W)
		W -= update
		if len(costs) % 100 == 0:
			print ".",
		if len(costs) > 2 and costs[-2] - costs[-1] < 0.0001:
			if alpha < 0.2:
				break
			else:
				alpha = alpha / 1.5
	print "."
	return W, costs[-1]

def predict(X, Y, W):
	pred = sigmoid(np.dot(X, W.T))
	for j in range(len(pred)):
		if pred[j] > 0.5:
			pred[j] = 1
		else:
			pred[j] = 0
	acc = (pred == Y).mean() * 100
	return acc

def plotData(X, Y):
	X1 = X[:, 0]
	X2 = X[:, 1]

	pos = (Y == 1)
	neg = (Y == 0)

	plt.plot(X1[pos], X2[pos], "r+", lw=2)
	plt.plot(X1[neg], X2[neg], "bo", lw=2)
	
def plotDecisionBoundary(X, Y, W):
	# if we are in two dimensions:
	if X.shape[1] < 3:
		plot_x = np.array([np.min(X[:,0]) - 2, np.max(X[:,0]) + 2])
		plot_y = (-1 / W[2]) * (W[1] * plot_x + W[0])

		plt.plot(plot_x, plot_y)

# OCR data
curr_dir = os.path.dirname(os.path.realpath(__file__))

data = mergeFiles(curr_dir)
X = data[:,:-1].astype(float)
Y = data[:,-1].astype(float)

# Toy problem
#data = np.loadtxt("ex2data1.txt", delimiter=",")
#X = data[:, 0:2]
#Y = data[:, 2]

# Reformating and adding intercept term
X_train = np.hstack((np.ones((len(X), 1)), X))
Y_train = Y.reshape(len(Y),1)

initial_theta = np.zeros((1,X_train.shape[1]))

options = {'full_output': True, 'maxiter': 5000}

W, cost, iterations, _, _ = opt.fmin(lambda t: cost_function(t, X_train, Y), initial_theta, **options)

print "final cost: ",cost

print "Accuracy with fmin: ", predict(X_train, Y, W), "%"
print "Number of iterations: ", iterations
if X.shape[1] < 3:
	plotData(X,Y)
	plotDecisionBoundary(X, Y, W)
	plt.show()

print "******************* Initializing gradient descent *******************"

W_grad, cost = train_gradient_descent(X_train, Y)

print "final cost: ", cost
print "Accuracy with gradient descent: ", predict(X_train, Y, W_grad), "%"
