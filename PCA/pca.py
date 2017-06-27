from __future__ import division
import numpy as np
from numpy import linalg
import math


# http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
def normalize(X):	
	mu = np.mean(X)
	return (X - mu) / (np.linalg.norm(X - mu)) 

def pca(X):
	# Initialize eigenvectors and eigenvalues
	U = np.zeros(len(X))
	S = np.zeros(len(X))
	# Normalize X
	X_norm = normalize(X)

	# Computing covariance matrix
	sigma = (1/len(X_norm)) * np.dot(X_norm.T, X_norm)
	# computing eigenvectors
	U, s, V = np.linalg.svd(sigma)

	# getting rid of an eigenvector with each iteration
	for k in range(U.shape[1])[::-1]:
		# If less than 99% of variance is retained, break loop
		variance_lost = 1 - np.sum(s[:k]) / np.sum(s)
		if variance_lost >= 0.01:
			break

	# extracting top k eigenvectors
	U_reduce = U[:, :k]
	# Project data
	z = np.dot(X_norm, U_reduce)
	# recover data
	x_approx = np.dot(z, U_reduce.T)

	return U_reduce, z, x_approx
