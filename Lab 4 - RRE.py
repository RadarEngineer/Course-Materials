# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 21:49:29 2020

@author: kyled
"""
import numpy as np
import matplotlib.pyplot as plt

""" Function - Radar Range Equation """ 

def rre(Pt, G, wavelength, rcs, R):
    Pr = (Pt * G**2 * wavelength**2 * rcs) / ((4*3.14)**3 * R**4)
    return Pr

def friss(Pt, Gt, Gr, wavelength, R):
    Pr = (Gt * Gr * Pt * wavelength**2) / ((4*3.14*R)**2)
    return Pr

""" Evaluate the RRE """

Pt = 10e3
G = 10**(32/10)
f = 10e9
wavelength = 3e8/f
rcs = 1000
R = np.linspace(5e3,100e3, num=501)
Pr = rre(Pt, G, wavelength, rcs, R)

Gt = 10**(15/10)
Pt = 50
Pj = friss(Pt, Gt, G, wavelength, R)


""" Plot the Results """
R = R/1e3
plt.figure(0)
plt.clf()
plt.plot(R, 10*np.log10(Pr/0.001), 'b-', R, 10*np.log10(Pj/0.001), 'r--')
plt.show()
plt.grid()
plt.xlim(0, R[-1])     # set the xlim to left, right
plt.ylim(-120, -20)     # set the xlim to left, right
plt.xlabel('range (km)')
plt.ylabel('received power (dBm)')

""" Plot the Results """
R = R/1e3
plt.figure(1)
plt.clf()
plt.plot(R, 10*np.log10(Pj/Pr), 'b-')
plt.show()
plt.grid()
plt.xlim(0, R[-1])     # set the xlim to left, right
#plt.ylim(-120, -20)     # set the xlim to left, right
plt.xlabel('range (km)')
plt.ylabel('JSR (dB)')