# SVM algorithm using simplified SMO
from __future__ import division
import cPickle, gzip
import numpy as np
from numpy import linalg
import time
import math


# encode labels such that 0 := -1
def encode_labels(y):
	y_encode = np.zeros(y.shape)
	for i in range(len(y)):
		y_encode[i] = -1 if (y[i] == 0) else 1
	return y_encode

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def gaussian_kernel(x,z, sigma = 0.5):
	return np.exp((-np.linalg.norm(x - z) ** 2) / (2 * sigma ** 2))

def linear_kernel(x,z):
	return np.dot(x, z.T)

class SVM(object):
	def __init__(self, kernel, inputs, C=1.0):
		self.kernel = kernel
		self.C = C

		self.alpha = np.zeros(inputs)
		self.alpha_tmp = np.zeros(inputs)
		self.b = 0

	def train(self, X, y, iterations = 10000, tol = 0.001):
		self.fit(X, y, iterations, tol)

		# alpha gives nan for certain values with gaussian kernel - normalizing until other solution found
		self.alpha = np.nan_to_num(self.alpha)

		self.w = self.compute_w(X, y)

		output = self.predict(X)
		accuracy = np.linalg.norm((output == y)) / len(X) * 100

		print "Accuracy is : ", accuracy, "%"

	def fit(self, X, y, iterations = 10000, tol = 0.001):
		# initializing graham matrix for X 
		self.K = self.compute_graham(X)
		self.error = np.zeros(len(X))
		self.w = np.zeros((X.shape[1], 1))
		b = 0

		for itr in range(iterations):
			self.changed_alphas = 0
			for i in range(len(X)):
				# computing error for input i and j
				self.error[i] = self.alpha.dot(y.T * self.K[i]) + b - y[i]

				if (self.error[i] * y[i] < -tol and self.alpha[i] < self.C) or (y[i] * self.error[i] > tol and self.alpha[i] > 0):				
					# Choose second distinct input at random and compute corresponding error
					j = i
					while i == j:
						j = np.random.randint(0, len(X) - 1)

					self.error[j] = self.alpha.dot(y.T * self.K[j]) + b - y[j]
					
					# saving old alpha
					self.alpha_tmp[i], self.alpha_tmp[j] = self.alpha[i], self.alpha[j]
					
					# computing bounds L and H such as L < alpha < H and 0 < alpha < C
					L, H = self.compute_L_H(i, j)

					# If L == H, skip to next iteration
					if L == H:
						continue

					eta = 2 * self.K[i][j] - self.K[i][i] - self.K[j][j]
					if eta >= 0:
						continue

					# Updating alpha[j] and clipping it in L, H boundaries
					self.alpha[j] += float(y[j] * (self.error[i] - self.error[j])) / eta

					if self.alpha[j] > H:
						self.alpha[j] = H
					elif self.alpha[j] < L:
						self.alpha[j] = L

					if abs(self.alpha[j] - self.alpha_tmp[j]) < 0.00001:
						continue

					# updating alpha[i]
					self.alpha[i] += y[i] * y[j] * (self.alpha_tmp[j] - self.alpha[j])

					# computing threshold b so as to satisfy KKT conditions
					b1 = b - self.error[i] - y[i] * (self.alpha[i] - self.alpha_tmp[i]) * self.K[i][i] - y[j] * (self.alpha[j] - self.alpha_tmp[j]) * self.K[i][j] 
					b2 = b - self.error[j] - y[i] * (self.alpha[i] - self.alpha_tmp[i]) * self.K[i][j] - y[j] * (self.alpha[j] - self.alpha_tmp[j]) * self.K[j][j] 
					if self.alpha[i] > 0 and self.alpha[i] < self.C:
						b = b1
					elif self.alpha[j] > 0 and self.alpha[j] < self.C:
						b = b2
					else:
						b = (b1 + b2) / 2

					self.changed_alphas += 1
				if self.changed_alphas == 0:
					k += 1
				else:
					k = 0
				print "alpha: ",self.alpha[j]
		return self.alpha

	def compute_w(self,X, y):
		return np.dot(y * self.alpha, X)

	def predict(self, X):
		return np.sign(np.dot(self.w.T, X.T) + self.b)

	def compute_L_H(self, i, j):
		if y[i] * y[j] > 0:
			L = max(0, self.alpha[j] - self.alpha[i])
			H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
		else:
			L = max(0, self.alpha[i] + self.alpha[j] - self.C)
			H = min(self.C, self.alpha[i] + self.alpha[j])
		return L, H

	def compute_graham(self, X):
		K = np.zeros((len(X), len(X)))
		for i in range(len(X)):
			for j in range(len(X)):
				K[i][j] = self.kernel(X[i], X[j])

		return K


data = np.loadtxt('ex2data1.txt', delimiter=',')

X = data[:, 0:4]
y = data[:, -1]
Y = encode_labels(y)


svm = SVM(linear_kernel, len(X))
svm.train(X, Y)
print svm.alpha


