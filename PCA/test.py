from __future__ import division
import numpy as np
from numpy import linalg
import math
import scipy.io
from pca import pca

mat = scipy.io.loadmat('ex7data1.mat')

X = mat['X']

U_reduce, z, x_approx = pca(X)


print "top eigenvectors", U_reduce[0, 0], U_reduce[1,0]
print "projection of first example", z[0]
print "Approx of first example", x_approx[0, 0], x_approx[0, 1]	