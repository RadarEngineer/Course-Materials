# -*- coding: utf-8 -*-
"""
Radar Training Data

@author: Kyle Davidson
Modified: 22 Sep 2019
"""
import matplotlib.pyplot as plt
import numpy as np
import math
from timeit import default_timer as timer

#------------------------------------------------------------------------------
# CLASS - Radar Waveform 
#------------------------------------------------------------------------------
class radar_waveform():
    
    def __init__(self, PRI, PW, modulation, BW, code, M, Ts):
        self.PRI = PRI    
        self.PW = PW
        self.modulation = modulation
        self.BW = BW
        self.Ts = Ts
        self.code = code
        self.M = M # Number of pulses in CPI
        self.alpha = 0.20 # Excess bandwidth in signal LPF (fraction)
        self.pol = 0 # Waveform polarization - linear (rad)
        
        self.create_pulse()
        #self.create_CPI()
     
    def create_pulse(self):
        
        if self.modulation == 'LFM':
            self.LFM()
        elif self.modulation == 'PSK':
            self.PSK()
        elif self.modulation == 'FSK':
            self.FSK()
        else:
            samples_in_pulse = int(math.ceil(self.PW/self.Ts))
            self.pulse = np.ones(samples_in_pulse, dtype ='float32')   
            
    def create_signal(self, stop):
        # Inserts the pulse into an array of zeros at the mid-point
        L = np.ceil(stop / self.Ts)
        L = L.astype(int)
        self.waveform = np.squeeze( np.zeros([1, L], dtype='float32') )
        
        idx = round(L/2 - np.size(self.pulse)/2)
        idx = idx.astype(int)
        self.waveform[idx:idx+np.size(self.pulse)] = self.pulse                
            
    def LFM(self):
        
        samples_in_pulse = math.ceil(self.PW/self.Ts)
        time = np.arange(0, samples_in_pulse*self.Ts, self.Ts, dtype ='complex64')
        k = self.BW/self.PW
        self.pulse = np.exp(1j*np.pi*k*time**2)
        
    def PSK(self):
        
        bits = self.code # Bit Sequence
        N = np.size(bits) # Number of Bits in the pulse   
        phase = np.pi/90 * bits # Convert bits from deg. to radians 
        phase = np.exp(1j*phase) # Convert to complex number format
        
        samples_in_bit = math.floor(self.PW/self.Ts/N)
        samples_in_pulse = samples_in_bit * N
        self.pulse = np.zeros(samples_in_pulse, dtype ='complex64')
        bit  = np.ones(samples_in_bit, dtype ='complex64')
        for k in range (0,N):
            self.pulse[(k)*samples_in_bit : (k+1)*samples_in_bit] = bit*phase[k]
        
    def FSK(self):
        
        freq = self.code
        M = np.size(freq) # Number of Bits in the pulse
        tb = self.PW/np.size(freq) # bit duration of the FSK bits
        fm = freq / tb
        samples_in_bit = math.ceil(self.PW/self.Ts/M)
        time = np.arange(0, samples_in_bit*self.Ts, self.Ts, dtype ='complex64')
        self.pulse = np.array([], dtype ='complex64')
        for k in range (0, M):
            bit = np.exp(1j*2*np.pi*fm[k]*time);
            self.pulse = np.concatenate((self.pulse,bit), axis=0);
                
#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------  
        
def add_noise(waveform, SNR):
    """ Adds WGN to the waveform at a signal-to noise ratio of SNR (dB) 
    Assumes a pulse amplitude of 1"""
    SNR = 10**(SNR/10) # convert to linear units (-)
    std_dev = np.sqrt(1/(2*SNR))
    noisy = waveform + std_dev * np.random.randn(np.size(waveform))
    noisy = noisy + 1j*std_dev * np.random.randn(np.size(waveform))
    return noisy        
    
#------------------------------------------------------------------------------
# Initialize
#------------------------------------------------------------------------------  

C = 0 # class counter
ts = 50e-9 # Sample period (s)
stop = 20e-6 # duration of sample time (s)

#------------------------------------------------------------------------------
# Demonstrate Waveform
#------------------------------------------------------------------------------
PRI = 500e-6 # Pulse repetition interval - set high to avoid mulitple pulses in pulse sample (s)
code = np.array([1, -1, 1], dtype = 'float32')
y = radar_waveform(PRI, 2e-6, 'LFM', 5e6, code, 4, ts)
y.create_signal(stop)
SNR = 30 # signal-to-noise ratio (dB)
y.waveform = add_noise(y.waveform , SNR)

y = y.waveform

t = np.arange(0, np.size(y)*ts, ts)

plt.figure(1)
plt.clf()
plt.plot(t*1e6, np.abs(y), 'm-.', label="Magnitude")
plt.plot(t*1e6, np.real(y), 'b', label="Real")
plt.plot(t*1e6, np.imag(y), 'r--', label="Imaginary") 
plt.axis([0, t[-1]*1e6, -1.5, 1.5])
plt.grid(True)
plt.xlabel('time (us)')
plt.ylabel('amplitude (V)')
plt.legend()
plt.show()


#------------------------------------------------------------------------------
# Create the training data for Unmodulated
#------------------------------------------------------------------------------
start1 = timer()
pulse_width = np.array([0.5, 1, 1.5, 2, 4, 5, 10], dtype = 'float32') * 1e-6 # Pulse width (s)

for idx in range(np.size(pulse_width)):
    if idx < 1:
        y = radar_waveform(PRI, pulse_width[idx], 'none', 5e6, code, 4, ts)
        y.create_signal(stop)
        x = np.zeros([np.size(pulse_width), np.size(y.waveform)], dtype='float32')
        x[idx,:] = y.waveform
    else:    
        y = radar_waveform(PRI, pulse_width[idx], 'none', 5e6, code, 4, ts)
        y.create_signal(stop)
        x[idx,:] = y.waveform
        
C = np.arange(np.size(x[:,1]))        
        
plt.figure(2)
plt.clf()
for idx in range(np.size(pulse_width)):
    plt.plot(t*1e6, np.real(x[idx,:])*(1+idx/10), 'b', label="Real")
plt.axis([0, t[-1]*1e6, -1.5, 2])
plt.grid(True)
plt.xlabel('time (us)')
plt.ylabel('amplitude (V)')
plt.show()

#------------------------------------------------------------------------------
# Create the training data for LFM
#------------------------------------------------------------------------------
bandwidth = np.array([1, 2, 3, 4, 5, 6, 8, 10, 15, 20], dtype = 'float32') * 1e6# Bandwidth (Hz)
pulse_width = np.array([0.5, 1, 1.5, 2, 4, 5, 10], dtype = 'float32') * 1e-6 # Pulse width (s)
print('\n Time elasped - waveform: ', (timer() - start1)*1e3, 'ms')





















