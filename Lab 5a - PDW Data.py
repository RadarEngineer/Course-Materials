# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 10:27:34 2020

@author: kyled
"""

import numpy as np
import matplotlib.pyplot as plt

""" CLASS - PDW """

class radar:
    def __init__(self, power, PW, freq, theta, phi, MOP, PMOP, FMOP, CW, PRI):
        self.power = power # Pulse amplitude (V)
        self.PW = PW # Pulse width (s)
        self.freq = freq # Pulse frequency (Hz)
        self.theta = theta # Azimuth (rad)
        self.phi = phi # Elevation (rad)
        self.MOP = MOP # Modulation on pulse (Yes/No)
        self.PMOP = PMOP # Phase modulation on pulse (Yes/No)
        self.FMOP = FMOP # Frequency modulation on pulse (Yes/No)
        self.CW = CW # Continuous wave (Yes/No)
        self.PRI = PRI # pulse repetition interval (s)
        
class pdw:
    def __init__(self):
        self.amplitude = 1 # Pulse amplitude (V)
        self.TOA = 0 # Pulse time of arrival (s)
        self.PW = 0 # Pulse width (s)
        self.freq = 0 # Pulse frequency (Hz)
        self.theta = 0 # Azimuth (rad)
        self.phi = 0 # Elevation (rad)
        self.MOP = 0 # Modulation on pulse (Yes/No)
        self.PMOP = 0 # Phase modulation on pulse (Yes/No)
        self.FMOP = 0 # Frequency modulation on pulse (Yes/No)
        self.CW = 0 # Continuous wave (Yes/No)
        #self.pol = 0 # Measured polarization (rad)

    def generate_random_pdw(self, freq_min, freq_max, pw_mean, pw_std, 
                            prob_PMOP, prob_FMOP, prob_CW, amp_mean, amp_std, t, PRI):
        # Generate the elevation with a Gaussian distribution of (mean, std_dev)
        self.amplitude = amp_std * np.random.randn(1) + amp_mean
        # Generate the azimuth with a uniform distribution between (0, 2*pi)
        self.theta = 2 * np.pi * np.random.rand(1)
        # Generate the elevation with a Gaussian distribution of (mean, std_dev)
        phi_std = 10*np.pi/180
        phi_mean = 0
        self.phi = phi_std * np.random.randn(1) + phi_mean
        # Generate the frequency with a uniform distribution between (fmin, fmax)
        self.freq = freq_min + (freq_max-freq_min) * np.random.rand(1)
        # Generate the pulse width with a Gaussian distribution between (mean, std_dev)
        self.PW = np.abs(pw_std * np.random.randn(1) + pw_mean)
        
        prob = np.random.rand(3)
        if prob[0] <= prob_PMOP:
            self.MOP = 1
            self.PMOP = 1
            self.FMOP = 0
        if prob[1] <= prob_FMOP:
            self.MOP = 1
            self.PMOP = 0
            self.FMOP = 1
        if prob[2] <= prob_CW:
            self.CW = 1
        
        self.TOA = t + PRI 

""" FUNCTION - Convert from Class to Array """    
    
def cls_to_arr(p):
    arr = np.zeros((10, len(p)))
    for n in range(0,len(p)):
        arr[0, n] = p[n].amplitude
        arr[1, n] = p[n].PW
        arr[2, n] = p[n].freq
        arr[3, n] = p[n].theta
        arr[4, n] = p[n].phi
        arr[5, n] = p[n].MOP
        arr[6, n] = p[n].PMOP
        arr[7, n] = p[n].FMOP
        arr[8, n] = p[n].CW
        arr[9, n] = p[n].TOA
        
    return arr

""" FUNCTION - Scale the data """    

def scale_data(p):
    # frequency
    # pulse width
    # azimuth, and elevation 
    return p

"""----------------------------------------------------------------------------
Main Program 
----------------------------------------------------------------------------"""

# Fixing random state for reproducibility
np.random.seed(2)
N = 10000 # Number of PDWs
M = 5 # Number of radars
p = [] # Array of PDWs

# Create radars
power = np.array([-40, -35, -55, -43, -50])
PW = np.array([1, 0.5, 2, 0.1, 1.2, 1])
freq = np.array([9.6, 8.4, 11.5, 10.2, 10])
theta = np.pi/180 * np.array([0, 20, 50, 270, 180])
phi = np.pi/180 * np.array([0, 0, 0, 0, 0])
MOP = np.array([0, 0, 0, 1, 1])
PMOP = np.array([0, 0, 0, 1, 1])
FMOP = np.array([0, 0, 0, 0, 1])
CW = np.array([0, 0, 0, 0, 1])
PRI = 1/ np.array([5000, 8000, 1000, 2000, 300])

# Create an array of radars
rad = []
for n in range(0,M):
    rad.append(radar(power[n], PW[n], freq[n], theta[n], phi[n], MOP[n], PMOP[n], FMOP[n], CW[n], PRI[n]))

#Create the array of PDWs
for n in range(0,N*M):
    p.append(pdw())

# Generate random data for the PDWs using the probability density functions
power_std = 1
m = 0
for m in range(0,len(rad)):
    t = 0 # timer for PRI
    for n in range(0,N):
        #p[n].generate_random_pdw(8, 12, 1, 0.5, 0.9, 0.9, 0.98, rad[m].power, power_std)
        p[n + (m)*N].generate_random_pdw(8, 12, rad[m].PW, 0.1, 0.9, 0.9, 0.98, rad[m].power, power_std, t, rad[m].PRI)
        t = t + rad[m].PRI
    
p = cls_to_arr(p) # convert to an array

# the histogram of the data
plt.figure(0)
plt.clf()
n, bins, patches = plt.hist(p[0,:], 200, density=True, facecolor='g', alpha=0.75)
plt.grid()
plt.show()
plt.xlabel('Received Power (dBm)')

plt.figure(1)
plt.clf()
n, bins, patches = plt.hist(p[1,:], 200, density=True, facecolor='g', alpha=0.75)
plt.grid()
plt.show()
plt.xlabel('Pulse Width ($\mu s$)')

plt.figure(2)
plt.clf()
n, bins, patches = plt.hist(p[2,:], 200, density=True, facecolor='g', alpha=0.75)
plt.grid()
plt.show()
plt.xlabel('Frequency (GHz)')





























