from __future__ import division
import numpy as np
import scipy.optimize as opt


def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigmoidPrime(z):
	return sigmoid(z)*(1-sigmoid(z))


class NeuralNetwork(object):

	def __init__(self, input, hidden, output):
		self.input = input + 1
		self.hidden = hidden
		self.output = output

		self.ai = [1.0] * self.input
		self.ah = [1.0] * self.hidden
		self.ao = [1.0] * self.output

		self.wi = np.random.randn(self.hidden, self.input)
		self.wo = np.random.randn(self.output, self.hidden)

		self.ci = np.zeros((self.input, self.hidden))
		self.co = np.zeros((self.hidden, self.output))

	def feedforward(self,X):
		# appending bias term x0 to X
		self.ai[1:] = X
		print self.ai
		# activation of hidden layer
		for j in range(len(self.ah)):
			for i in range(len(self.ai)):
				self.ah[j] = sigmoid(self.ai[i] * self.wi[j][i])

		# activation of output layer
		for k in range(len(self.ao)):
			for j in range(len(self.ah)):
				self.ao[k] = sigmoid(self.ah[j] * self.wo[k][j])

		return self.ao[:]

	def backpropagation(self, y, epsilon):
		# computing error for ao
		delta_o = self.ao - y

		delta_h = [0] * len(self.ah)
		# computing error for ah
		for j in range(len(self.ah)):
			for i in range(len(self.ao)):
				delta_h[j] = self.wo[i][j] * delta_o[i] * sigmoidPrime(self.wo[i][j] * self.ah[j])


		# update weights
		for i in range(self.output):
			for j in range(self.hidden):
				change = self.ah[j] * delta_o[i] 
				self.wo[i][j] -= epsilon * change + self.co[j][i]
				self.co[j][i] = change

		for j in range(self.hidden):
			for k in range(self.input):
				change = self.ai[k] * delta_h[j]
				self.wi[k][j] -= epsilon * change + self.ci[j][k]
				change = change

		error = 0.0
		for i in range(len(self.ao)):
			error += 0.5 * (self.ao[i] - y) ** 2
		return error

	def train(self, X, y, iterations = 10000, epsilon = 0.00001):

		for i in range(iterations):
			error = 0.0
			for k in range(X.shape[0]):
				x = X[k]
				Y = y[k]
				self.feedforward(x)
				error = self.backpropagation(Y, epsilon)
			
			if i % 100 == 0:
				print "error: %-.5f" % error

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:,0:2]
y = data[:,2]
nn = NeuralNetwork(2,3,2)

nn.feedforward(X[1])
nn.backpropagation(y[1],0.5)

nn.train(X,y)
print nn.feedforward(X[3])
print y[3]
