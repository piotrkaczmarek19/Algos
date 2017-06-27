# first working logistic regression
from __future__ import division
import numpy as np
import scipy.optimize as opt
import cPickle, gzip
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

with gzip.open('mnist.pkl.gz', 'rb') as f:
	train_set, valid_set, test_set = cPickle.load(f)

print 'Shapes'
print '\tTraining: ', train_set[0].shape, train_set[1].shape
print '\tValidation: ', valid_set[0].shape, valid_set[1].shape
print '\tTest: ', test_set[0].shape, test_set[1].shape
print '\tLabels: ', np.unique(train_set[1])

def sigmoid(z):
	return 1 / (1+np.exp(-z))

def encode_labels(y):
	Y = np.zeros((y.shape[0], 10))

	for i in range(len(y)):
		Y[i][y[i]] = 1

	return Y

def costFunction(x,Y,W, epsilon=0.1):
	reg = epsilon * (W * W).mean()
	J =  -Y*np.log(sigmoid(x.dot(W.transpose()))) - (1-Y)*np.log(1-sigmoid(x.dot(W.transpose())))
	return J.mean() + reg

def costGrad(x,Y,W, epsilon=0.1):
	reg = epsilon * W
	grad = (sigmoid(x.dot(W.transpose())) - Y).transpose().dot(x)
	return (grad + reg) / len(y) 

def train(x,Y,W, alpha = 0.01, iterations=1000):
	costs = []
	for i in range(iterations):	
		costs.append(float(costFunction(x,Y,W)))
		update = alpha * costGrad(x,Y,W)
		W -= update
		if len(costs) % 10 == 0:
			print 'Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha
		if len(costs) > 2 and costs[-2] - costs[-1] < 0.0001:
			if alpha < 0.2:
				break
			else:
				alpha = alpha / 1.5

	#plt.plot(range(len(costs)), costs)
	#plt.title('Cost vs training epoch')
	#plt.show()

	predictions = sigmoid(x.dot(W.transpose()))
	accuracy = np.sum(np.rint(np.logical_and((x.dot(W.transpose())), Y))) / len(Y) * 100
	print "\tAccuracy: ", accuracy, "%"

	activations = [W[i, :].reshape((28, 28)) for i in xrange(W.shape[0])]

	for i, w in enumerate(activations):
		plt.subplot(5, 10, i + 1)
		plt.set_cmap('gray')
		plt.axis('off')
		plt.imshow(w)
	plt.show()

x = train_set[0]
y = train_set[1]
labels = len(np.unique(y))
Y = encode_labels(y)
# Initialize weights

W = np.random.random((labels,x.shape[1])) * 0.1


print 
train(x,Y,W)



