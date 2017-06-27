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

with gzip.open('mnist.pkl.gz', 'rb') as f:
	train_set, valid_set, test_set = cPickle.load(f)

print 'Shapes'
print '\tTraining: ', train_set[0].shape, train_set[1].shape
print '\tValidation: ', valid_set[0].shape, valid_set[1].shape
print '\tTest: ', test_set[0].shape, test_set[1].shape
print '\tLabels: ', np.unique(train_set[1])


W_shape = (10, 784)
b_shape = 10

# shared - theano constructor-> creates reusable variable
W = shared(np.random.random(W_shape) - 0.5, name="W")
b = shared(np.random.random(b_shape) - 0.5, name="b")

x = T.dmatrix("x") # N x 784
labels = T.dmatrix("labels") # N x 10

output = T.nnet.softmax(x.dot(W.transpose()) + b)
prediction = T.argmax(output, axis=1)
cost = T.nnet.binary_crossentropy(output, labels).mean()

reg_lambda = 0.01
regularized_cost = cost + reg_lambda * ((W * W).sum() + (b * b).sum())

compute_prediction = function([x], prediction)
compute_cost = function([x, labels], cost)

grad_W = grad(regularized_cost, W)
grad_b = grad(regularized_cost, b)

alpha = T.dscalar("alpha")
updates = [(W, W - alpha * grad_W),(b,b - alpha * grad_b)]

train_regularized = function([x, labels, alpha], regularized_cost, updates=updates)

alpha = 10
labeled = encode_labels(train_set[1], 9)

costs = []

while True:
	costs.append(float(train_regularized(train_set[0], labeled, alpha)))

	if len(costs) % 10 == 0 :
		print 'Epoch ', len(costs), 'with cost ', costs[-1], 'and alpha', alpha
	if len(costs) > 2 and costs[-2] - costs[-1] < 0.0001:
		if alpha < 0.2:
			break
		else:
			alpha = alpha / 1.5

prediction = compute_prediction(test_set[0])

def accuracy(predicted, actual):
	total = 0.0
	correct = 0.0
	# zip returns tuples of merged sequences (for i,j in zip(range(K), range(M)))
	for p,a in zip(predicted, actual):
		total += 1
		if p == a:
			correct += 1
	return correct / total

print accuracy(prediction, test_set[1])

val_W = W.get_value()
activations = [val_W[i,:].reshape((28,28)) for i in xrange(val_W.shape[0])]

print activations[0].shape


for i,w in enumerate(activations):
	plt.subplot(1,10, i+1)
	plt.set_cmap('gray')
	plt.axis('off')
	plt.imshow(w)
	plt.show()
plt.gcf().set_size_inches(9,9)

