# Image compression algorithm (B&W)
from __future__ import division
import numpy as np
from numpy import linalg
import math
import matplotlib
import matplotlib.pyplot as plt
from scipy import misc
from kmeans import *

# saving image f as filename 'test.png'
#misc.imsave('test.png', f)

# Loading image into a three dimensionnal array
filename = 'js'
image = misc.imread('js_flat.png')
img_size =  image.shape
print img_size
# reshaping image into N X 3 matrix where N = number of pixels; each row contains rbg pixel values
X = image.reshape(img_size[0] * img_size[1], img_size[2])
# Divide by 255 so that all values are in the range 0 - 1
X = X / 255
# Initialize parameters
k = 32
max_iter = 10

# get centroids and labels
centroids, labels = kmeans(X, k, max_iter)
# get final labels into form list
idx = list(getLabels(X, centroids))

# 
X_recovered = centroids[idx] 
img_compressed = X_recovered.reshape(img_size[0], img_size[1], img_size[2])

filename_new = filename+'_compressed.png' 
misc.imsave(filename_new, img_compressed)
