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



# Transmitted and received signals for channel estimation and tests
fs = 44100

sync_chirp_duration = 1
sync_chirp = gen_chirp(sync_chirp_duration, form = 'linear')

wn_duration = 1
wn = np.random.randn(wn_duration*fs)

test_sig_duration = sync_chirp_duration + wn_duration
test_sig = np.concatenate((sync_chirp, wn))
write('channel_h_test_signal.wav', fs, test_sig)


print('RECORDING...')
received_sig = sd.rec(int(3*test_sig_duration*fs), samplerate = fs, channels = 1, blocking = True).flatten()
print('RECORDING FINISHED')

# Synchronise to find the start of the chirp in the received signal
convolution = np.convolve(received_sig, np.flip(sync_chirp))
chirp_start_index = np.abs(convolution).argmax() - sync_chirp_duration*fs

# Truncate the received signal to isolate the chirp and the whole test signal
chirp_end_index = np.minimum(int(chirp_start_index + sync_chirp_duration*fs), received_sig.size)
received_chirp = received_sig[chirp_start_index:chirp_end_index]

sig_end_index = np.minimum(int(chirp_start_index + 1.2*test_sig_duration*fs), received_sig.size)
received_sig = received_sig[chirp_start_index:sig_end_index]



# FrFT search to find the channel impulse response
# Don't search from α = 0: The optimum FrFT will never be at α = 0 but sometimes the largest l-inf. norm of the FrFT is at α = 0, causing errors in the channel impulse response
alpha_values = np.arange(0.01, 2, 0.01)

Y_max_array = []
for alpha in alpha_values:
	# Find the FrFT of the received chirp at given alpha
	FrFT = frft(received_chirp, alpha)

	# Finds the element of the FrFT with the largest magnitude
	Y_max = np.linalg.norm(FrFT, ord = np.inf)

	Y_max_array.append(Y_max)

Y_max_array = np.asarray(Y_max_array)

# Optimum value of alpha is the one that gives the FrFT with the largest l-infinity norm
alpha_opt = alpha_values[np.argmax(Y_max_array)]
Y_alpha_opt = frft(received_chirp, alpha_opt)

# Estimate the channel impulse response
frft_power, frft_extent = get_spectrogram_data(Y_alpha_opt, 1.2*test_sig_duration, fs)
h_dc_power = frft_power[int(frft_power.shape[0]/2)][0:int(0.1*fs)]		# Estimate from the DC signal power

h_frft = np.abs(Y_alpha_opt)[0:int(0.1*fs)]		# Estimate from the magnitude of Y_alpha_opt directly



# Calculate the estimated received signals using each estimate of the channel impulse response
estimated_sig_dcsigpower = np.convolve(test_sig, h_dc_power)
estimated_sig_frft = np.convolve(test_sig, h_frft)

# Plot results to compare the actual and estimated received signals
fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2, 3, figsize = (18, 12))

fig.suptitle('Channel Estimation Test Results: $α_{opt}$ = '+str(np.round(alpha_opt, 2)))

# Actual received signal
rc_power, rc_extent = get_spectrogram_data(received_sig, 1.2*test_sig_duration, fs)
rc_im = ax0.imshow(10*np.log10(rc_power), aspect = 'auto', interpolation = None, origin = 'lower', extent = rc_extent)
rc_cbar = plt.colorbar(rc_im, ax = ax0)
rc_cbar.set_label('Signal Power [dB]')
ax0.set_title('Received Chirp')
ax0.set_xlabel('Time [s]')
ax0.set_ylabel('Freqeuncy [Hz]')

# Estimated signal spectrogram using DC signal powers
sig_est_dc_power, sig_est_dc_extent = get_spectrogram_data(estimated_sig_dcsigpower, 1.2*test_sig_duration, fs)
sig_est_dc_im = ax1.imshow(10*np.log10(sig_est_dc_power), aspect = 'auto', interpolation = None, origin = 'lower', extent = sig_est_dc_extent)
sig_est_dc_cbar = plt.colorbar(sig_est_dc_im, ax = ax1)
sig_est_dc_cbar.set_label('Signal Power [dB]')
ax1.set_title('Estimated Received Chirp: DC Signal Power')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Freqeuncy [Hz]')

# Estimated signal spectrogram using Y_alpha_opt
sig_est_frft_power, sig_est_frft_extent = get_spectrogram_data(estimated_sig_dcsigpower, 1.2*test_sig_duration, fs)
sig_est_frft_im = ax2.imshow(10*np.log10(sig_est_frft_power), aspect = 'auto', interpolation = None, origin = 'lower', extent = sig_est_frft_extent)
sig_est_frft_cbar = plt.colorbar(sig_est_dc_im, ax = ax2)
sig_est_frft_cbar.set_label('Signal Power [dB]')
ax2.set_title('Estimated Received Chirp: FrFT')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Freqeuncy [Hz]')

# FrFT chirp spectrogram
frft_im = ax3.imshow(10*np.log10(frft_power), aspect = 'auto', interpolation = None, origin = 'lower', extent = frft_extent)
frft_cbar = plt.colorbar(frft_im, ax = ax3)
frft_cbar.set_label('Signal Power [dB]')
ax3.set_title('FrFT Chirp at α = {}'.format(alpha_opt))
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Freqeuncy [Hz]')

# Impulse response from DC signal power
h_ts = np.arange(0, 0.1, 1/fs)
#channel_ts = np.linspace(frft_extent[0], frft_extent[1], frft_power.shape[1])
ax4.plot(h_ts, h_dc_power, color = 'blue')
ax4.set_title('Channel Impulse Response: DC Signal Power')
ax4.set_xlabel('Time [s]')
ax4.set_ylabel('Channel Impulse Response')

# Impulse response from Y_alpha_opt
#gamma = np.var(np.abs(Y_alpha_opt[:int(0.01*fs)]))		# Noise floor estimated using parts of the FrFT containing no parts of the impulse response
gamma = 0.1*np.amax(np.abs(Y_alpha_opt))
ax5.plot(h_ts, h_frft, color = 'blue')
ax5.axhline(gamma, color = 'red', linestyle = ':', label = 'Noise floor γ = {}'.format(np.round(gamma, 3)))
ax5.set_title('Channel Impulse Response: FrFT at α = {}'.format(alpha_opt))
ax5.set_xlabel('Time [s]')
ax5.set_ylabel('Channel Impulse Response')
ax5.legend(loc = 'upper right')

plt.show()