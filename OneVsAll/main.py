import numpy as np
import pylab as py
import scipy.optimize as opt
import scipy.io
import random
import time
from scipy.misc import toimage

def sigmoid(A):
	G = 1/(1+np.exp(-A))

	return G

def costFunction(theta,X,y,alpha):
	m = len(X)
	#Regularization
	theta_reg = theta
	theta_reg[1] = 0
	
	h = sigmoid(X.dot(theta))
	l = -y*np.log(h)-(1-y)*np.log(1-h) 

	return l.mean() - alpha * np.sum(theta_reg ** 2)

def grad(theta,X,y,alpha):
	m = len(X)
	n = theta.shape[0]
	h = sigmoid(X.dot(theta))
	theta_reg = theta
	theta_reg[1] = 0

	grad = (1/m)*(h - y).dot(X)-alpha/m * theta_reg

	return grad


# Train num_labels successive logistic regressions for all_theta and 
# keep label that maximizes value of X * all_theta
def oneVsAll(X,y,num_labels,lbda):
	m = X.shape[0]
	n = X.shape[1]

	all_theta = np.zeros((num_labels,n+1))
	# Add intercept terms
	X = np.append(np.ones((m,1)),X,axis=1)
	theta = 0.1*np.random.randn(n+1,)

	c = 0
	while c < num_labels:
		theta = opt.fmin_cg(costFunction,theta,fprime=grad,args=(X,(y==c),lbda))		
		all_theta[c][:] = theta
		c +=1
	
	return all_theta

def predictOneVsAll(all_theta,X, num_labels):
	m = X.shape[0]
	p = np.zeros((m,1))

	X = np.append(np.ones((m,1)),X,axis=1)

	predict = sigmoid(X.dot(all_theta.T))



def main(data):
	data = scipy.io.loadmat(data)
	y = data[data.keys()[0]]
	X = data[data.keys()[1]]

	# 20X20 Input Images of Digits
	input_layer_size = X.shape[1]
	num_labels = int(max(y))-int(min(y))+1
	# Training set size & regularization parameter
	m = X.shape[0]
	lbda = 0.1

	all_theta = oneVsAll(X,y,num_labels,lbda)	
	#pred = predictOneVsAll(all_theta,X, num_labels)


if __name__ == "__main__":
	main("ex3data1.mat")

	