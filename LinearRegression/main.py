import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import random
import time

def generateSample(N, variance=100):
    X = np.matrix(range(N)).T + 1
    Y = np.matrix([random.random() * variance + i * 10 + 900 for i in range(len(X))]).T
    return X, Y

# Mini Batch Gradient (becomes stochastic if gradientStep == 1)
def miniBatch_gradient(x, y, gradientStep=1):
    m = len(x)
    theta = np.zeros((x.shape[1], 1))
    eta = 0.0001

    maxIteration = 10000
    for i in range(0,maxIteration):
        k = 0
        l = gradientStep
        while l <= m:  
            error = x[k:l] * theta - y[k:l]
            gradient = x[k:l].T * error
            theta = theta - eta * gradient
            k += gradientStep
            l += gradientStep
        #Cleaning up if m not multiple of gradient Step
        if l<m:
            error = x[l:m] * theta - y[l:m]
            gradient = x[l:m].T * error
            theta = theta - eta * gradient
    return theta

def plotModel(x, y, w):
    plt.plot(x[:,1], y, "x")
    plt.plot(x[:,1], x * w, "r-")
    plt.show()

def test(N, variance, modelFunction,step):
    X, Y = generateSample(N, variance)
    X = np.hstack([np.matrix(np.ones(len(X))).T, X])
    w = modelFunction(X, Y,step)
    plotModel(X, Y, w)
    return True


if __name__ == '__main__':
    start = time.time()
    #test(50, 600, miniBatch_gradient,1)
    #print("Time for stochastic %f" % (time.time()  - start))
    #start = time.time()
    test(50, 600, miniBatch_gradient,10)
    print("Time for minibatch %f" % (time.time() - start))