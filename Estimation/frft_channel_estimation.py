"""
Test for channel estimation using the Fractional Fourier Transform (FrFT)
Source: https://ieeexplore.ieee.org/abstract/document/9221028
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
import struct, wave, frft
import soundfile as sf
import sounddevice as sd
from utilities import *



# Finds the start index of the received chirp by convolving with the time-reversed known chirp
# Returns the start index of the chirp in the unknown signal array
def get_chirp_end_index(sent_chirp, received_chirp):
	convolution = np.convolve(received_chirp, np.flip(sent_chirp))
	c = np.abs(convolution)

	plt.plot(c)
	plt.show()

	signal_end_index = c.argmax()

	return signal_end_index



# Transmitted and received chirps for channel estimation
fs = 44100
chirp_f_start = 10
chirp_f_end = 20000
chirp_duration = 2
time_samples = np.arange(0, chirp_duration, 1/fs)

test_chirp = chirp(time_samples, chirp_f_start, chirp_duration, chirp_f_end, method = 'linear')
write('frft_test_chirp.wav', fs, test_chirp)

print('RECORDING...')
received_chirp = sd.rec(int(2.5*chirp_duration*fs), samplerate = fs, channels = 1, blocking = True).flatten()
print('RECORDING FINISHED')

# Synchronise to find the start of the chirp in the received sample
chirp_start_index = get_chirp_end_index(test_chirp, received_chirp) - chirp_duration*fs

# Plot the received signal and the estimated start index
plt.plot(received_chirp, color = 'blue')
plt.axvline(chirp_start_index, ymin = 0, ymax = 1, color = 'red')
plt.show()



# Implement FrFT search
received_chirp_trunc = received_chirp[chirp_start_index:]
alpha_values = np.arange(0, 2, 0.1)

amp = np.abs(received_chirp_trunc)
phs = np.angle(received_chirp_trunc)

# Complex-valued object of received chirp
obj_1d = amp * np.exp(1.j * phs)
obj_1d_shifted = np.fft.fftshift(obj_1d)

Y_max_array = []
for alpha in alpha_values:
	# Find the FrFT of the received chirp at given alpha
	fobj_1d = frft.frft(obj_1d_shifted, alpha)
	FrFT = np.fft.fftshift(fobj_1d)

	# Finds the element of the FrFT with the largest magnitude
	Y_max = np.linalg.norm(FrFT, ord = inf)
	
	Y_max_array.append()

Y_max_array = np.asarray(Y_max_array)

# Optimum value of alpha is the one that gives the FrFT with the largest l-infinity norm
alpha_opt = alpha_values[np.argmax(Y_max_array)]

# Finds the FrFT at the optimum value of alpha
fobj_1d = frft.frft(obj_1d_shifted, alpha_opt)
Y_alpha_opt = np.fft.fftshift(fobj_1d)




# Estimate the channel using the FrFT
# Noise floor of the recording
gamma = np.var(Y_alpha_opt[:100])

# Estimate the channel coefficients as the FrFT values greater than the noise floor
h_estimate = np.zeros(Y_alpha_opt.size)
for i, val in enumerate(Y_alpha_opt):
	if val > gamma:
		h_estimate[i] = val