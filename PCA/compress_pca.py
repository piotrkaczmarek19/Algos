 # -*- coding: utf-8 -*-
import numpy as np
from numpy import linalg
from pca import pca
from scipy import misc
import pylab as pyl
import scipy.io

mat = scipy.io.loadmat('ex7faces.mat')

X = mat['X']

image = misc.imread('bird_small.png')
img_size = image.shape
print img_size
X = np.mean(image,2)


U_reduce, z, x_approx = pca(X)

print "size of X: ",X.shape
print "size of z: ", z.shape
pyl.imshow(x_approx)
pyl.show()
