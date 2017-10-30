 #Reject method for computation of standard normal samples by samples from Cauchy law

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
#import plotly.plotly as py
from math import sqrt, exp, pi    #from smthg import * means we import all the functions of smthg and we can use them without specify smthg. before
import numpy as np

N = int(input("Number of samples ? "))			#Number of samples we wantinput to ask the user input() return a string, int change it into an int

# Fct for the reject condition f/2g f~N(0,1) and g~Cauchy
def rejectfct(val):
    val = ((1/sqrt(2*pi))*exp(-val**2/2))/(2*(1/pi)*(1/(val**2+1)))
    return val

# Main fct of reject method, parameter M is the number of samples we want
def rejectmethod(M):
    result = np.array([])				#Empty list to store the selected samples

    for k in range(M):
        unif = 1
        tres = 0
        while unif > tres:
            cauchy = np.random.standard_cauchy(1)
            unif = np.random.random_sample(1)
            tres = rejectfct(Cauchy[0])
            result = np.append(result,cauchy[0])
    return result

R = rejectmethod(N)

print(R)

# rejectmethod seems to works but we can build a histogram to check that

# Trying to build the histogram by hand : bad idea
"""def checkreject(x):
    nreal = 39
    B = np.zeros(nreal)
    I = np.arange(-10,10.5,0.5)
    for i in range(nreal):
        B[i] = np.sum((x>I[i]) & (x<I[i+1]))
    return B
"""

# Trying to use the built-in fcts of matplotlib : better idea but need to
# understand what is going on here
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
"""
