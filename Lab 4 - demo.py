# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:10:10 2020

@author: kyled
"""

import numpy as np
import matplotlib.pyplot as plt


""" Function for the Radar Range Equation """

def rre(Pt, G, wavelength, rcs, R):
    # Calculates the power received (W)
    Pr = (Pt * G**2 * wavelength**2 * rcs) / ((4*3.14)**3 * R**4)
    return Pr


""" Function - Friss """
def friss(Pt, Gt, Gr, wavelength, R):
    Pr = (Gt * Gr * Pt * wavelength**2) / ((4*3.14*R)**2)
    return Pr

""" Main Program """

Pt = 10e3; # Transmit power (W)
G = 10**(32/10) # Antenna gain (-)
f = 10e9 # frequency (Hz)
wavelength = 3e8/f # wavelength (m)
rcs = 1000 # Radar cross section (m^2)

R = np.linspace(5e3, 100e3, num=501);

Pr = rre(Pt, G, wavelength, rcs, R)

Gj = 10**(15/10) # Antenna gain of jammer (-)
Ptj = 50 # Transmit Power of jammer (W)
Pj = friss(Ptj, Gj, G, wavelength, R)

""" Plot the Results """ 

R = R/1e3 # convert the range from (m) to (km)

plt.figure(0)
plt.clf()
plt.plot(R, 10*np.log10(Pr/0.001), 'b-', label='radar')
plt.plot(R, 10*np.log10(Pj/0.001), 'r--', label='jammer')
plt.grid()
plt.xlabel('range (km)')
plt.ylabel('power (dBm)')
plt.xlim(0, R[-1])
plt.ylim(-120, -20)
plt.legend()
plt.show()

""" Plot the JSR """

plt.figure(1)
plt.clf()
plt.plot(R, 10*np.log10(Pj/Pr), 'b-', label='JSR')
plt.grid()
plt.xlabel('range (km)')
plt.ylabel('power (dBm)')
plt.xlim(0, R[-1])
plt.ylim(10, 50)
plt.show()

