"""
Program to test the channel coefficents estimated from the FrFT of the linear chirp
Currently uses a block of white noise immediately after the chirp for the test
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import scipy.signal as sig
import sounddevice as sd
from utilities import *
import frft



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



# Transmitted and received signals for channel estimation and tests
fs = 48000
chirp_f_start = 1000
chirp_f_end = 10000
chirp_duration = 1

chirp_ts = np.arange(0, chirp_duration, 1/fs)
chirp = sig.chirp(chirp_ts, chirp_f_start, chirp_duration, chirp_f_end, method = 'linear')

wn_duration = 1
wn = np.random.randn(wn_duration*fs)

test_sig_duration = chirp_duration + wn_duration
test_sig = np.concatenate((chirp, wn))
write('channel_h_test_signal.wav', fs, test_sig)


print('RECORDING...')
received_sig = sd.rec(int(3*test_sig_duration*fs), samplerate = fs, channels = 1, blocking = True).flatten()
print('RECORDING FINISHED')

# Synchronise to find the start of the chirp in the received signal
convolution = sig.convolve(received_sig, np.flip(chirp))
chirp_start_index = np.abs(convolution).argmax() - chirp_duration*fs

# Truncate the received signal to isolate the chirp and the whole test signal
chirp_end_index = int(chirp_start_index + chirp_duration*fs)
sig_end_index = int(chirp_start_index + test_sig_duration*fs)

received_chirp = received_sig[chirp_start_index:chirp_end_index]
received_sig = received_sig[chirp_start_index:sig_end_index]



# FrFT search to find the channel impulse response
a_opt = frft.optimise_a(received_chirp).x
Y_opt = frft.frft(received_chirp, a_opt)

# Correct the time scaling of the impulse response from the FrFT
time_axes = np.arange(0, len(Y_opt), 1/np.cos(a_opt))
frft_axes = np.arange(0, len(Y_opt))
Y_opt = np.interp(time_axes, frft_axes, Y_opt)

# Estimate the channel impulse response
h_start_index = np.argmax(Y_opt)
h = Y_opt[h_start_index:h_start_index+8192]
h[:int(0.001*fs)] = 0

h_ts = np.arange(0, h.size/fs, 1/fs)

gamma = np.var(np.abs(h))
#h[np.abs(h) < gamma] = 0 		# Filter for noise



# Calculate the estimated received signals using each estimate of the channel impulse response
estimated_sig = sig.convolve(test_sig, h)

# Plot results to compare the actual and estimated received signals
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize = (18, 12))

fig.suptitle('Channel Estimation Test Results: $α_{opt}$ = '+str(np.round(a_opt, 2)))

# Actual received signal
rc_power, rc_extent = get_spectrogram_data(received_sig, test_sig_duration, fs)
rc_im = ax0.imshow(10*np.log10(rc_power), aspect = 'auto', interpolation = None, origin = 'lower', extent = rc_extent)
rc_cbar = plt.colorbar(rc_im, ax = ax0)
rc_cbar.set_label('Signal Power [dB]')
ax0.set_title('Received Chirp')
ax0.set_xlabel('Time [s]')
ax0.set_ylabel('Freqeuncy [Hz]')

# Estimated signal spectrogram using Y_opt
estimated_sig_power, estimated_sig_extent = get_spectrogram_data(estimated_sig, test_sig_duration, fs)
estimated_sig_im = ax1.imshow(10*np.log10(estimated_sig_power), aspect = 'auto', interpolation = None, origin = 'lower', extent = estimated_sig_extent)
estimated_sig_cbar = plt.colorbar(estimated_sig_im, ax = ax1)
estimated_sig_cbar.set_label('Signal Power [dB]')
ax1.set_title('Estimated Received Chirp')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Freqeuncy [Hz]')

# Estimated impulse response
ax2.plot(h_ts, np.abs(h), color = 'blue')
ax2.axhline(gamma, color = 'red', linestyle = ':', label = 'Noise floor γ = {}'.format(np.round(gamma, 3)))
ax2.set_title('Channel Impulse Response')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('|h(t)|')
ax2.legend(loc = 'upper right')

# Estimated frequency response
H = np.fft.fft(h)
H_freqs = np.linspace(0, fs, H.size)
ax3.plot(H_freqs, 10*np.log10(np.abs(H)), color = 'blue')
ax3.set_title('Channel Frequency Response')
ax3.set_xlabel('Frequency [Hz]')
ax3.set_ylabel('|H(f)| [dB]')

plt.show()