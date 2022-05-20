"""
Test for channel estimation using the Fractional Fourier Transform (FrFT)
Source: https://ieeexplore.ieee.org/abstract/document/9221028
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read
import scipy.signal as sig
import struct, wave, tftb
import soundfile as sf
import sounddevice as sd
from utilities import *



# FrFT function from https://programtalk.com/vs2/python/10788/pyoptools/pyoptools/misc/frft/frft.py/
def frft(x, alpha):
    assert x.ndim == 1, 'x must be a 1 dimensional array'
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

# Function to get the info needed the plot the spectrogram of a signal
def get_spectrogram_data(signal, T, fs = 44100):
	dt = 1/fs	# timestep
	num_samples = T*fs
	ts = np.arange(num_samples)/fs

	# Looking at the power of the short time fourier transform (SFTF):
	nperseg = 2**6  # window size of the STFT
	f_stft, t_stft, Zxx = sig.stft(signal, fs, nperseg=nperseg, noverlap=nperseg-1, return_onesided=False)

	# Shift the frequency axis for better representation
	Zxx = np.fft.fftshift(Zxx, axes=0)
	f_stft = np.fft.fftshift(f_stft)

	# Plot the spectogram
	df = f_stft[1] - f_stft[0]  # the frequency step

	signal_power = np.real(Zxx * np.conj(Zxx))
	extent = ts[0] - dt/2, ts[-1] + dt/2, f_stft[0] - df/2, f_stft[-1] + df/2
	#im = plt.imshow(np.real(Zxx * np.conj(Zxx)), aspect='auto', interpolation=None, origin='lower', extent=extent)

	return signal_power, extent



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

chirp_start_index = np.abs(convolution).argmax() - chirp_duration*fs

# Truncate the received signal to only include the chirp
chirp_end_index = np.minimum(int(chirp_start_index + 1.2*chirp_duration*fs), received_chirp.size)
received_chirp_trunc = received_chirp[chirp_start_index:chirp_end_index]



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

Y_alpha_opt = frft(received_chirp_trunc, alpha_opt)
print('FrFT at α_opt is: {}'.format(Y_alpha_opt))

"""
# Recommended method from the referenced paper, but this gives very strange values for the noise floor
gamma = np.var(np.abs(Y_alpha_opt))
print('Noise floor γ = {}'.format(gamma))

# Estimate the channel coefficients as the FrFT values greater than the noise floor
h_estimate = np.zeros(Y_alpha_opt.size, dtype = complex)
for i, val in enumerate(Y_alpha_opt):
	if np.abs(val) > gamma:
		h_estimate[i] = val
"""



# Plot spectrograms of the received and FrFT'd chirps
fig, (ax0, ax1) = plt.subplots(1, 2, figsize = (12, 6))

fig.suptitle('Channel Estimation Test Results')

rc_power, rc_extent = get_spectrogram_data(received_chirp_trunc, 1.2*chirp_duration, fs)
rc_im = ax0.imshow(rc_power, aspect = 'auto', interpolation = None, origin = 'lower', extent = rc_extent)
plt.colorbar(rc_im, ax = ax0)
ax0.set_title('Received Chirp Spectrogram')
ax0.set_xlabel('Time [s]')
ax0.set_ylabel('Freqeuncy [Hz]')

frft_power, frft_extent = get_spectrogram_data(Y_alpha_opt, 1.2*chirp_duration, fs)
frft_im = ax1.imshow(frft_power, aspect = 'auto', interpolation = None, origin = 'lower', extent = frft_extent)
plt.colorbar(frft_im, ax = ax1)
ax1.set_title('FrFT Chirp Spectrogram')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Freqeuncy [Hz]')

plt.show()