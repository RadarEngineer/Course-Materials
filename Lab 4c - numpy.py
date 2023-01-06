# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 18:58:59 2020

@author: kyled
"""
import numpy as np
import time

a = np.array([[1,2,3],[4,5,6]])
print(a.shape)
print(a.dtype)

# change the dtype to 'float64' 
a = a.astype('float64')
print(a.dtype)

""" Time for float32 """ 
a = np.array([[1,2,3],[4,5,6]], dtype='float32')
b = np.array([[1,2,3],[4,5,6]], dtype='float32')
t = time.time()
for idx in range(1000):
    c = a + b;

print(time.time() - t)

""" Single Array Operations """
arr = np.arange(0,9,1)
print(arr)
arr = np.zeros([3,3])
print(arr)
arr = np.identity(3)
print(arr)

""" Multiple Array Operations """ 
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.concatenate((a,b), axis = 0)
print(c)
c = np.convolve(a,b)
print(c)
c = np.dot(a,b)
print(c)
c = np.cross(a,b)
print(c)