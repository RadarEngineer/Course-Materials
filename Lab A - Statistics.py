# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:23:39 2020

@author: Kyle Davidson
"""

# Here we're importing the Numpy library
import numpy as np
from matplotlib import pyplot as plt

""" Maxima and Minima """
a = np.array([[3,7,5],[8,4,3],[2,4,9]]) 

print('Our array is:')
print(a)
print('\n')

print('Applying amin() function:')
print(np.amin(a,1)) 
print('\n')

print('Applying amin() function again:')
print(np.amin(a,0)) 
print('\n')

print('Applying amax() function:')
print(np.amax(a)) 
print('\n')

print('Applying amax() function again:')
print(np.amax(a, axis = 0))
print('\n')

""" Median """

a = np.array([[30,65,70],[80,95,10],[50,90,60]]) 

print('Our array is:')
print(a)
print('\n') 

print('Applying median() function:')
print(np.median(a))
print('\n')

print('Applying median() function along axis 0:')
print(np.median(a, axis = 0))
print('\n')
 
print('Applying median() function along axis 1:')
print(np.median(a, axis = 1))
print('\n')


""" Mean """

import numpy as np 
a = np.array([[1,2,3],[3,4,5],[4,5,6]]) 

print('Our array is:')
print(a)
print('\n')   

print('Applying mean() function:')
print(np.mean(a))
print('\n')  

print('Applying mean() function along axis 0:')
print(np.mean(a, axis = 0))
print('\n')  

print('Applying mean() function along axis 1:' )
print(np.mean(a, axis = 1))
print('\n')

""" Variance """

print(np.std([1,2,3,4]))
print('\n')
print(np.var([1,2,3,4]))
print('\n')

""" Normal Distributions """ 

mu, sigma = 10, 0.1 # mean and standard deviation
x = np.random.normal(mu, sigma, 10000)

print('Verify the mean and the variance:')
print(abs(mu - np.mean(x)));
# 0.0  # may vary
print(abs(sigma - np.std(x, ddof=1)))
# 0.1  # may vary

plt.figure(0)
plt.clf()
count, bins, ignored = plt.hist(x, 50, density=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
plt.grid()
plt.show()


""" Uniform Distribution """ 

x = np.random.uniform(-1,0,10000)
# All values are within the given interval:

np.all(x >= -1)
np.all(x < 0)

#Display the histogram of the samples, along with the probability density function:

plt.figure(1)
plt.clf()
count, bins, ignored = plt.hist(x, 50, normed=True)
plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
plt.grid()
plt.show()


""" Curve Fitting """

x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
z = np.polyfit(x, y, 3)


p = np.poly1d(z)
p30 = np.poly1d(np.polyfit(x, y, 30))
    
    
xp = np.linspace(-2, 6, 100)
plt.figure(2)
plt.clf()
_ = plt.plot(x, y, '.', xp, p(xp), '-', xp, p30(xp), '--')
plt.ylim(-2,2)
plt.xlim(0,6)
plt.grid()
plt.show()




























