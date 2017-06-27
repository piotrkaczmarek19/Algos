# http://andrew.gibiansky.com/blog/machine-learning/coding-intro-to-nns/
from __future__ import division
import numpy as np
import scipy.optimize as opt
from theano import *
import theano.tensor as T
import cPickle, gzip
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1 / (1+np.exp(-z))

def dsigmoid(z):
	return z * (1.0-z)

def encode_labels(labels, max_index):
	encoded = np.zeros((labels.shape[0], max_index + 1))

	for i in xrange(labels.shape[0]):
		encoded[i, labels[i]] = 1
	return encoded


def accuracy(predicted, actual):
	total = 0.0
	correct = 0.0
	for p, a in zip(predicted, actual):
		total += 1
		if p == a:
			correct += 1
	return correct / total

with gzip.open('mnist.pkl.gz', 'rb') as f:
	train_set, valid_set, test_set = cPickle.load(f)

print 'Shapes'
print '\tTraining: ', train_set[0].shape, train_set[1].shape
print '\tValidation: ', valid_set[0].shape, valid_set[1].shape
print '\tTest: ', test_set[0].shape, test_set[1].shape
print '\tLabels: ', np.unique(train_set[1])


# Initialize shared weights variables
W1_shape = (50, 784)
b1_shape = 50
W2_shape = (10,50)
b2_shape = 10

W1 = shared(np.random.random(W1_shape) - 0.5, name="W1")
b1 = shared(np.random.random(b1_shape) - 0.5, name="b1")
W2 = shared(np.random.random(W2_shape) - 0.5, name="W2")
b2 = shared(np.random.random(b2_shape) - 0.5, name="b2")

# symbolic inputs
x = T.dmatrix("x") # N x 784
labels = T.dmatrix("labels") # N x 10

# Symbolic outputs
hidden = T.nnet.sigmoid(x.dot(W1.transpose()) + b1)
output = T.nnet.softmax(hidden.dot(W2.transpose()) + b2)
prediction = T.argmax(output, axis=1)
reg_lambda =  0.0001
regularization = reg_lambda * ((W1 * W1).sum() + (W2 * W2).sum() + (b1 * b1).sum() + (b2 * b2).sum())
cost = T.nnet.binary_crossentropy(output, labels).mean() + regularization

compute_prediction = function([x], prediction)

alpha = T.dscalar("alpha")
weights = [W1, W2, b1, b2]
updates = [(w, w - alpha * grad(cost, w)) for w in weights]
train_nn = function([x, labels, alpha], cost, updates=updates)

alpha = 10.0
labeled = encode_labels(train_set[1], 9)

costs = []
while True:
	costs.append(float(train_nn(train_set[0], labeled, alpha)))

	if len(costs) % 10 == 0:
		print 'Epoch', len(costs), 'with cost', costs[-1], 'and alpha', alpha
	if len(costs) > 2 and costs[-2] - costs[-1] < 0.0001:
		if alpha < 0.2:
			break
		else:
			alpha = alpha / 1.5


prediction = compute_prediction(test_set[0])
print accuracy(prediction, test_set[1])

val_W1 = W1.get_value()
activations = [val_W1[i, :].reshape((28, 28)) for i in xrange(val_W1.shape[0])]

for i, w in enumerate(activations):
	plt.subplot(5, 10, i + 1)
	plt.set_cmap('gray')
	plt.axis('off')
	plt.imshow(w)

plt.subplots_adjust(hspace=-0.85)
plt.gcf().set_size_inches(9, 9)
plt.show()