# nn with use of fmin_cg
from __future__ import division
import numpy as np
import scipy.optimize
import cPickle, gzip
import matplotlib
import matplotlib.pyplot as plt
import time
import Image

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

	def feedforward(self, wi, wo):
		self.ah = sigmoid(self.ai.dot(wi.transpose()) + self.bi)

		self.ao = sigmoid(self.ah.dot(wo.transpose()) + self.bo)

		return self.ao		

	def costFunction(self, nn_params, lambda_reg=0.3):
		# reshaping weights
		wi = nn_params[:self.input * self.hidden].reshape((self.hidden,self.input))
		wo = nn_params[self.input * self.hidden:].reshape((self.output,self.hidden))

		self.feedforward(wi, wo)

		J = -np.sum(Y * np.log(self.ao) + (1 - Y) * np.log(1 - self.ao))/len(self.ai)

		reg = lambda_reg / (2*len(self.ai)) * (np.sum(wi*wi) + np.sum(wo*wo) + (self.bi * self.bi).sum() + (self.bo * self.bo).sum())

		return J + reg

	def gradCost(self, nn_params, lambda_reg=0.3):
		# reshaping weights
		wi = nn_params[:self.input * self.hidden].reshape((self.hidden,self.input))
		wo = nn_params[self.input * self.hidden:].reshape((self.output,self.hidden))

		delta_o = self.ao - self.Y

		delta_h = np.dot(delta_o, self.wo) * dsigmoid(self.ah)	

		wo_grad = lambda_reg * np.transpose(delta_o).dot(self.ah) / len(Y)
		wi_grad = lambda_reg * np.transpose(delta_h).dot(self.ai) / len(Y)

		# return unravelled grad
		return np.append(wi_grad,wo_grad)	

	def train(self, X, Y):
		# unravelled weights and initializing input layers
		nn_params = np.append(self.wi,self.wo)
		self.Y = Y
		self.ai = X	
		self.ah = np.ones((len(X), self.hidden))
		self.ao = np.ones((len(X), self.output))

		scipy.optimize.fmin_cg(self.costFunction, nn_params, self.gradCost, maxiter=1000)

		print "Accuracy: ", self.predict(self.ao,Y), "%"

	def predict(self,output, Y):
		pred = np.zeros(Y.shape)
		for i in xrange(output.shape[0]):
			for j in xrange(output.shape[1]):
				pred[i][j] = 1 if output[i][j]>0.5 else 0
		return np.sum(np.logical_and(pred, Y)) / len(Y) * 100

start = time.time()

nn = Neural_NetworkMLP(784,300,10)

X = train_set[0][:15000]
Y = encode_labels(train_set[1][:15000])

nn.train(X, Y)

print "time: ",time.time() - start, "s"

nn.ai = valid_set[0]
Y_val = encode_labels(valid_set[1])

output = nn.feedforward(nn.wi, nn.wo)
print "Accuracy on validation set: ", nn.predict(output,Y_val), "%"


for i in range(20,35):
	X_test = test_set[0][i]
	y_test = test_set[1][i]

	nn.ai = X_test
	output_test = np.argmax(nn.feedforward(nn.wi, nn.wo))

	print "Actual classifictation for test example: ", y_test 
	print "Predicted output for test", output_test



#nn.ai = data

#print "Prediction for example image: ", nn.feedforward(nn.wo)