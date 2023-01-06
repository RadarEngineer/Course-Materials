# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:29:38 2020

@author: kyled
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 20:12:07 2020

@author: kyled
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# x from 0 to 30
N = 50
x = 30 * np.random.random((N, 1))

# y = a*x + b with noise
y = 0.5 * x + 1.0 + np.random.normal(size=x.shape)

# create a linear regression model
model = LinearRegression()
model.fit(x, y)

# predict y from the data
x_new = np.linspace(0, 30, 100)
y_new = model.predict(x_new[:, np.newaxis])

# plot the results
plt.figure(figsize=(4, 3))
ax = plt.axes()
ax.scatter(x, y)
ax.plot(x, 0.5 * x + 1.0, linestyle='-', color='blue', label="line2")
ax.plot(x_new, y_new, linestyle='--', color='red', label="line2")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.axis('tight')
plt.show()
plt.grid()