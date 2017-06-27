# based on https://pythonprogramming.net/support-vector-machine-svm-example-tutorial-scikit-learn-python/
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import cPickle, gzip
import numpy as np
import time

with gzip.open('mnist.pkl.gz', 'rb') as f:
	train_set, valid_set, test_set = cPickle.load(f)

print 'Shapes'
print '\tTraining: ', train_set[0].shape, train_set[1].shape
print '\tValidation: ', valid_set[0].shape, valid_set[1].shape
print '\tTest: ', test_set[0].shape, test_set[1].shape
print '\tLabels: ', np.unique(train_set[1])


#digits = datasets.load_digits()

#print(digits.data)
#print digits.target

clf = svm.SVC(gamma=0.01, C=100)
X,y = train_set[0], train_set[1]

start = time.time()
clf.fit(X,y)

pred = clf.predict(X)
accuracy =  clf.score(X,y)

print "Accuracy training set: ", accuracy 
print "Runnning time: ", time.time() - start, "s"
#print(plt.imshow(digits.images[-8], cmap=plt.cm.gray_r, interpolation='nearest'))
#plt.show()

