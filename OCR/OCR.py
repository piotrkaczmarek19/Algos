# OCR optimization using sklearn
from __future__ import division
import os
import numpy as np 
from processData import mergeFiles
from sklearn import linear_model

curr_dir = os.path.dirname(os.path.realpath(__file__))

data = mergeFiles(curr_dir)
X = data[:,:-1].astype(float)
Y = data[:,-1].astype(float)


# Reformating and adding intercept term
X_train = np.hstack((np.ones((len(X), 1)), X))
Y_train = Y.reshape(len(Y),1)

lr = linear_model.LogisticRegression(C=10)

lr.fit(X, Y)
print lr
pred = lr.predict(data[:,:-1])

print "accuracy: ",(pred == Y).mean() * 100, "%"