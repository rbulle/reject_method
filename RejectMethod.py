 #Reject method for computation of standard normal samples by samples from Cauchy law

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#import plotly.plotly as py
from math import sqrt, exp, pi    #from smthg import * means we import all the functions of smthg and we can use them without specify smthg. before
import numpy as np

#from pdb import set_trace
N = input("Number of samples ? ")			#Number of samples we want input to ask the user
N = int(N)			#input() return a string, int change N into an integer

def rejectfct(val):			#Function for the reject condition f/2g
    val = ((1/sqrt(2*pi))*exp(-val**2/2))/(2*(1/pi)*(1/(val**2+1)))
    return val

def rejectmethod(M):
    X = np.array([])				#Empty list to store the selected samples

    #set_trace()
    for k in range(M):
        U = 1
        tres = 0
        while U > tres:
            Y = np.random.standard_cauchy(1)
            U = np.random.random_sample(1)
            tres = rejectfct(Y[0])
        X=np.append(X,Y[0])
    return X

R = rejectmethod(N)

"""def checkreject(x):
    nreal = 39
    B = np.zeros(nreal)
    I = np.arange(-10,10.5,0.5)
    for i in range(nreal):
        B[i] = np.sum((x>I[i]) & (x<I[i+1]))
    return B
"""

plt.hist(R, normed=True, bins=30)
plt.title("Standard Gaussian hist")
plt.xlabel("Value")
plt.ylabel("Frequency")

fig = plt.gcf()

bins = 30
y = mlab.normpdf(bins, 0, 1)
plt.plot(bins, y, 'r--')

plt.subplots_adjust(left=0.15)
plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')

