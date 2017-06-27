#Importing libraries. The same will be used throughout the article.
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression

rcParams['figure.figsize'] = 12, 10

#Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducability
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.plot(data['x'],data['y'],'.')

for i in range(2,16):  #power of 1 is already there
	colname = 'x_%d'%i #new var will be x_power
	data[colname] = data['x']**i
print data.head()