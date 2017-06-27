# first working nn
from __future__ import division
import numpy as np
import scipy.optimize as opt
from theano import *
import theano.tensor as T
import cPickle, gzip
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import time

with gzip.open('mnist.pkl.gz', 'rb') as f:
	train_set, valid_set, test_set = cPickle.load(f)

print 'Shapes'
print '\tTraining: ', train_set[0].shape, train_set[1].shape
print '\tValidation: ', valid_set[0].shape, valid_set[1].shape
print '\tTest: ', test_set[0].shape, test_set[1].shape
print '\tLabels: ', np.unique(train_set[1])

def sigmoid(z):
	return 1 / (1+np.exp(-z))

def dsigmoid(z):
	return sigmoid(z) * (1-sigmoid(z))

def encode_labels(y):
	Y = np.zeros((y.shape[0], 10))

	for i in range(len(y)):
		Y[i][y[i]] = 1

	return Y

class Neural_NetworkMLP(object):

	def __init__(self, input, hidden, output):
		self.input = input
		self.hidden = hidden
		self.output = output

		self.wi = np.random.random((self.hidden,self.input)) - 0.5
		self.bi = np.random.random(self.hidden) - 0.5
		self.wo = np.random.random((self.output, self.hidden)) - 0.5
		self.bo = np.random.random(self.output) - 0.5
 
		self.ci = np.zeros((self.hidden, self.input))
		self.co = np.zeros((self.output, self.hidden))
	def feedforward(self, X):
		self.ai = X

		self.ah = sigmoid(X.dot(self.wi.transpose()) + self.bi)

		self.ao = sigmoid(self.ah.dot(self.wo.transpose()) + self.bo)

		return self.ao

	def backpropagation(self, Y, alpha, reg_lambda=1):
		delta_o = self.ao - Y

		delta_h = np.dot(delta_o, self.wo) * dsigmoid(self.ah)

		# update (10, 50) (5000,10) (5000,50)
		change = np.transpose(delta_o).dot(self.ah)
		self.wo -= alpha * change/len(Y)
		self.co = change + self.co

		change = np.transpose(delta_h).dot(self.ai) 
		self.wi -= alpha * change/len(Y) + self.ci
		self.ci = change 

		reg = reg_lambda * ((self.wi * self.wi).sum() + (self.wo * self.wo).sum() + (self.bi * self.bi).sum() + (self.bo * self.bo).sum())
		cost = 1/2 * np.sum(delta_o ** 2) 
		return cost

	def train(self, X, Y, alpha = 1, iterations=700):
		costs = []
		for i in range(iterations):	
			self.feedforward(X)
			cost = self.backpropagation(Y, 0.5, 0.1)
			costs.append(float(cost))

			if len(costs) % 10 == 0:
				print 'Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha

		print "Accuracy: ", self.predict(Y), "%"
		plt.plot(range(len(costs)), costs)
		plt.title('Cost vs training epoch')
		plt.show()


	def predict(self, Y):
		pred = np.zeros(Y.shape)
		for i in xrange(Y.shape[0]):
			for j in xrange(Y.shape[1]):
				pred[i][j] = 1 if self.ao[i][j]>0.5 else 0
		return np.sum(np.logical_and(pred, Y)) / len(Y) * 100


nn = Neural_NetworkMLP(784,100,10)
X = train_set[0]
Y = encode_labels(train_set[1])

start = time.time()

nn.train(X,Y)

print "time: ",time.time() - start,"s"

