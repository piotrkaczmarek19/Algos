# K-means algorithm - base code: http://stanford.edu/~cpiech/cs221/handouts/kmeans.html
from __future__ import division
import numpy as np
from numpy import linalg
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import scipy.io

def kmeans(X, k, max_iter):
	np.seterr(divide='ignore', invalid='ignore')
	numFeatures = X.shape[1]
	inputs = X.shape[0]
	centroids = chooseRandomCentroids(X, k)

	iterations = 0
	oldCentroids = None

	# Iterate while iter < max_iter or centroids converge
	while iterations < max_iter and not (np.array_equal(oldCentroids, centroids)):
		print "Kmeans iteration ", iterations," out of ", max_iter

		oldCentroids = centroids
		iterations += 1

		# assign a centroid to each training example
		labels = getLabels(X, centroids)

		# get new centroids using labels previously acquired
		centroids = getCentroids(X, labels, k) 
	return centroids, labels

def chooseRandomCentroids(X, k):
	# picking k training examples at random and return them as centroids
	idx = np.random.randint(0,len(X),k)

	return X[idx]

def getLabels(X, centroids):
	# final array of labels for each training example
	c = np.zeros(len(X))
	# norm of each training example to each centroid to be stored in cache
	temp = np.zeros(len(centroids))

	for i in range(len(X)):
		for j in range(len(centroids)):
			temp[j] = np.linalg.norm(X[i] - centroids[j])
		# centroids with minimal norm to training example becomes assigned to that training example
		c[i] = np.argmin(temp)	
	return c

def getCentroids(X, labels, k):
	# generating array of k points
	centroids = np.zeros((k,X.shape[1]))
	for j in range(k):
		# extract training examples assigned to current centroid j
		X_j = X[labels == j]
		centroids[j] = np.mean(X_j)
	return centroids

