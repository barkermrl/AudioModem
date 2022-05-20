"""
Test for channel estimation using the Fractional Fourier Transform (FrFT)
Source: https://ieeexplore.ieee.org/abstract/document/9221028
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
import struct, wave
import soundfile as sf
import sounddevice as sd
from utilities import *



# FrFT function from https://programtalk.com/vs2/python/10788/pyoptools/pyoptools/misc/frft/frft.py/
def frft(x, alpha):
    assert x.ndim == 1, "x must be a 1 dimensional array"
    m = x.shape[0]
    p = m #m-1 # deveria incrementarse el sigiente pow -- 'Should increase the following pow'
    y = np.zeros((2*p,), dtype = complex)
    z = np.zeros((2*p,), dtype = complex)
     
    j = np.indices(z.shape)[0]
    y[0:m] = x*np.exp(-1.j*np.pi*(j[0:m]**2)*float(alpha)/m)
     
     
    z[0:m] = np.exp(1.j*np.pi*(j[0:m]**2)*float(alpha)/m)
    z[-m:] = np.exp(1.j*np.pi*((j[-m:]-2*p)**2)*float(alpha)/m)
     
     
    d = np.exp(-1.j*np.pi*j**2**float(alpha)/m)*np.fft.ifft(np.fft.fft(y)*np.fft.fft(z))
     
    return d[0:m]



# Transmitted and received chirps for channel estimation
fs = 44100
chirp_f_start = 20
chirp_f_end = 20000
chirp_duration = 1
time_samples = np.arange(0, chirp_duration, 1/fs)

test_chirp = chirp(time_samples, chirp_f_start, chirp_duration, chirp_f_end, method = 'linear')
write('frft_test_chirp.wav', fs, test_chirp)

print('RECORDING...')
received_chirp = sd.rec(int(3*chirp_duration*fs), samplerate = fs, channels = 1, blocking = True).flatten()
print('RECORDING FINISHED')

# Synchronise to find the start of the chirp in the received sample
convolution = np.convolve(received_chirp, np.flip(test_chirp))

chirp_end_index = np.abs(convolution).argmax()
chirp_start_index = chirp_end_index - chirp_duration*fs

received_chirp_trunc = received_chirp[chirp_start_index:]
assert received_chirp_trunc.size >= test_chirp.size



# FrFT search
alpha_values = np.arange(0, 2, 0.05)

Y_max_array = []
for alpha in alpha_values:
	# Find the FrFT of the received chirp at given alpha
	FrFT = frft(received_chirp_trunc, alpha)

	# Finds the element of the FrFT with the largest magnitude
	Y_max = np.linalg.norm(FrFT, ord = np.inf)

	Y_max_array.append(Y_max)

Y_max_array = np.asarray(Y_max_array)

# Optimum value of alpha is the one that gives the FrFT with the largest l-infinity norm
alpha_opt = alpha_values[np.argmax(Y_max_array)]
print('α_opt = {}'.format(alpha_opt))

# Finds the FrFT at the optimum value of alpha
Y_alpha_opt = frft(received_chirp_trunc, alpha_opt)
print('FrFT at α_opt is: {}'.format(Y_alpha_opt))




# Estimate the channel using the FrFT
"""
# Recommended method from the referenced paper, but this gives very strange values for the noise estimate
# Noise floor of the recording, currently estimating as a percentage of the max. amplitude of the FrFT. By eye, 5% seems like a good value
gamma = np.var(np.abs(Y_alpha_opt))
print('Noise floor γ = {}'.format(gamma))

# Estimate the channel coefficients as the FrFT values greater than the noise floor
h_estimate = np.zeros(Y_alpha_opt.size, dtype = complex)
for i, val in enumerate(Y_alpha_opt):
	if np.abs(val) > gamma:
		h_estimate[i] = val
"""

# Estimate the channel coefficients by smoothing using a rolling average over a given time duration
smoothing_duration = 0.1		# in seconds
kernel_size = int(smoothing_duration*fs)
kernel = np.ones(kernel_size)/kernel_size
h_estimate = np.convolve(Y_alpha_opt, kernel, mode = 'same')



# Plot the transmitted and received chirps (from the estimated chirp start index) and the estimated channel coefficients
fig, axs = plt.subplot_mosaic([['ax0', 'ax1'], ['ax2', 'ax2']])
fig = plt.figure(figsize = (10, 6), constrained_layout = True)
spec = fig.add_gridspec(2, 2)
ax0 = fig.add_subplot(spec[0, 0])
ax1 = fig.add_subplot(spec[0, 1])
ax2 = fig.add_subplot(spec[1, :])

fig.suptitle('Channel Estimation Test Results')

ax0.plot(np.abs(convolution), color = 'blue')
ax0.set_title('Chirp Synchronisation')
ax0.set_xlabel('Time Index')
ax0.set_ylabel('Convolution Amplitude')

received_chirp_trunc_time_samples = np.linspace(0, received_chirp_trunc.size/fs, received_chirp_trunc.size)

ax1.plot(time_samples, 10*np.log10(np.abs(test_chirp)), color = 'blue', label = 'Transmitted chirp')
ax1.plot(received_chirp_trunc_time_samples, 10*np.log10(np.abs(received_chirp_trunc)), color = 'red', label = 'Received chirp')
ax1.set_title('Chirp Signals')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Signal Power (dB)')
ax1.legend(loc = 'upper right')

channel_coefficient_time_samples = np.linspace(0, Y_alpha_opt.size/fs, Y_alpha_opt.size)

ax2.plot(channel_coefficient_time_samples, np.abs(h_estimate), color = 'blue', label = 'Channel coefficients')
#ax2.axhline(gamma, color = 'red', linestyle = ':', label = 'Noise floor γ')
ax2.set_title('Channel Response: Optimum FrFT α = {}'.format(np.round(alpha_opt, 2)))
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Channel Coefficient Magnitudes (Smoothed)')
ax2.legend(loc = 'upper right')

plt.show()