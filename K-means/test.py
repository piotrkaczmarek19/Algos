# Test on a two dimensional problem with three classes
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
from kmeans import kmeans


mat = scipy.io.loadmat('ex7data2.mat')

X = mat['X']

centroids, labels =  kmeans(X, 3, 10)

X_2 = X[labels == 2]
X_1 = X[labels == 1]
X_0 = X[labels == 0]
plt.scatter(X_2[:,0],X_2[:,1], color='y')
plt.scatter(X_1[:,0],X_1[:,1], color='r')
plt.scatter(X_0[:,0],X_0[:,1], color='b')

plt.show()