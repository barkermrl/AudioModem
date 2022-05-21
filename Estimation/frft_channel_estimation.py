"""
Test for channel estimation using the Fractional Fourier Transform (FrFT)
Source: https://ieeexplore.ieee.org/abstract/document/9221028

Current standard for transmition is:
1s pause, 2*1s chirp, 1s pause, 1s white noise
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import scipy.signal as sig
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
alpha_values = np.arange(0.01, 2, 0.01)

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
Y_alpha_opt = frft(received_chirp_trunc, alpha_opt)



# Chanel coefficient estimation from the FrFT
"""
# Recommended method for channel coefficient estimation, but this gives very strange values for the noise floor
gamma = np.var(np.abs(Y_alpha_opt))
print('Noise floor γ = {}'.format(gamma))

# Estimate the channel coefficients as the FrFT values greater than the noise floor
h_estimate = np.zeros(Y_alpha_opt.size, dtype = complex)
for i, val in enumerate(Y_alpha_opt):
	if np.abs(val) > gamma:
		h_estimate[i] = val
"""



# Current plan is to go through the FrFT chirp spectogram values at frequency f = 0 and sample the signal powers at each time
# From this, we can probably get a good idea of what the channel response looks like, as well as time delays between peaks
# We expect to see slightly broad peaks due to slight path length differences around each echo of the chirp
frft_power, frft_extent = get_spectrogram_data(Y_alpha_opt, 1.2*chirp_duration, fs)
# frft_power contains a matrix where each value contains the signal power at that time-frequency element
# The rows contain the time samples at given frequencies; row 0 is at +fs/2, row -1 is at -fs/2
dc_sig_powers = frft_power[int(frft_power.shape[0]/2)]


# Plot spectrograms of the received and FrFT'd chirps and the estimated channel coefficients
fig, axd = plt.subplot_mosaic([['ax0', 'ax1'], ['ax2', 'ax2']])
fig = plt.figure(figsize=(12, 8), constrained_layout=True)
spec = fig.add_gridspec(2, 2)
ax0 = fig.add_subplot(spec[0, 0])
ax1 = fig.add_subplot(spec[0, 1])
ax2 = fig.add_subplot(spec[1, :])

fig.suptitle('Channel Estimation Test Results')

rc_power, rc_extent = get_spectrogram_data(received_chirp_trunc, 1.2*chirp_duration, fs)
rc_im = ax0.imshow(rc_power, aspect = 'auto', interpolation = None, origin = 'lower', extent = rc_extent)
plt.colorbar(rc_im, ax = ax0)
ax0.set_title('Received Chirp Spectrogram')
ax0.set_xlabel('Time [s]')
ax0.set_ylabel('Freqeuncy [Hz]')

frft_im = ax1.imshow(frft_power, aspect = 'auto', interpolation = None, origin = 'lower', extent = frft_extent)
plt.colorbar(frft_im, ax = ax1)
ax1.set_title('FrFT Chirp Spectrogram: α = {}'.format(alpha_opt))
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Freqeuncy [Hz]')

channel_ts = np.linspace(frft_extent[0], frft_extent[1], frft_power.shape[1])
ax2.plot(channel_ts, dc_sig_powers, color = 'blue')
ax2.set_title('Estimated Channel Coefficients')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('DC Signal Power')

plt.show()