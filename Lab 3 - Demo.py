# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:12:13 2020

@author: kyled
"""

import numpy as np
import matplotlib.pyplot as plt

""" Function - Update Paramters """

def update_parameters(x, y, w, b, alpha):
    # initializing the parameters
    dl_dw = 0.0
    dl_db = 0.0
    
    # Calculate the partial derivatives
    for i in range(N):
        dl_dw += -2*x[i]*(y[i]-(w*x[i] + b))
        dl_db += -2*(y[i]-(w*x[i] + b))  
        
    # Update the parameters, w and b
    w = w - (1/float(N))*dl_dw*alpha
    b = b - (1/float(N))*dl_db*alpha
    return w, b

""" Function - Train Parameters """

def train(x, y, w, b, alpha, epochs):
    
    for e in range(epochs):
        w, b = update_parameters(x,y,w,b, alpha)
    
        # log the progress
        if e % 400 == 0:
            print("epoch:", e, "loss: ", avg_loss(x, y, w, b))
    
    return w, b

""" Function - Average Loss """
def avg_loss(x, y, w, b):
    N = len(x)
    total_error = 0.0 
    
    for i in range(N):
        total_error += (x[i]-(w*y[i] + b))**2

    return total_error / float(N)

""" Main Program """

# Create scenario data
N = 100
mu, sigma = 0, 2
y0 = 5
noise = np.random.normal(mu, sigma, N)
x = np.linspace(0,10, num = N)
y = x + noise + y0

# Plot the results
fig, ax = plt.subplots()
ax.plot(x, y, linestyle='-', label="line1")
plt.show()
plt.grid()