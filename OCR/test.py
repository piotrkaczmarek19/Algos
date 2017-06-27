from skimage import io, segmentation as seg
import pylab
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
from skimage import data, io
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import matplotlib.pyplot as plt
import scipy





img = io.imread('bird_small.png')

img = rgb2gray(img)
print img.shape
# returns evenly spaced numbers over a specified interval
center_img_x = img.shape[0] / 2
center_img_y = img.shape[1] / 4
s = np.linspace(0, 2*np.pi, 400)
x = center_img_x + 68*np.cos(s)
y = center_img_y + 68*np.sin(s)
init = np.array([x, y]).T

snake = active_contour(gaussian(img, 3), init, alpha = 0.015, beta=10, gamma=0.001)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
plt.gray()
ax.imshow(img)
ax.plot(init[:, 0], init[:, 1], '--r', lw=3)
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

print snake.shape

plt.show()