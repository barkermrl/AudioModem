# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import sounddevice as sd
from utilities import *



# Define functions

# Returns the middle index of a synchronisation chirp
def synchronise(received_signal, transmitted_chirp):
    def window(x): return x * np.hanning(len(x))
    
    c = sig.convolve(window(received_signal), np.flip(window(transmitted_chirp)), mode='same')

    return np.argmax(np.abs(c))

# Returns a channel estimate from the known and received OFDM symbols
def get_channel_estimate(known_symbol, received_symbols, num_symbols):
    
    R = np.zeros((num_symbols, known_symbol.shape[0]), dtype=complex)
    
    for i in range(num_symbols):
        r = received_symbols[L+i*(N+L):(i+1)*(L+N)]
        R[i,:] = np.fft.fft(r)

    H_est = np.mean(R, axis = 0)/known_symbol
    
    return H_est

# Plots a channel's frequency response and phase
def plot_channel(H, fs = 48000):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize = (15, 5))

    freqs = np.linspace(0, fs, H.shape[0])

    ax_left.plot(freqs, 10*np.log10(np.abs(H)))
    ax_left.set_xlabel('Frequency [Hz]')
    ax_left.set_ylabel('Magnitude [dB]')

    ax_right.scatter(freqs, np.angle(H))
    ax_right.set_xlabel('Frequency [Hz]')
    ax_right.set_ylabel('Phase [rad]')
    ax_right.set_ylim(-np.pi, np.pi)

    plt.show()


# Plots decoded symbols coloured by frequency bin
def plot_decoded_symbols(X):
	plt.scatter(X.real, X.imag, c = np.arange(X.shape[0]), cmap = 'gist_rainbow_r', marker = '.')

	plt.title('Decoded constellation symbols')
	plt.xlabel('Re')
	plt.ylabel('Im')
	cbar = plt.colorbar()
	cbar.set_label('Symbol number')

	plt.show()



# Define parameters

fs = 48000  	# Sampling frequency
N = 8192    	# OFDM symbol length
L = 1024    	# Cyclic prefix length

# constellation_map = {
#     '00' : complex(1/np.sqrt(2), 1/np.sqrt(2)),
#     '01' : complex(-1/np.sqrt(2), 1/np.sqrt(2)),
#     '11' : complex(-1/np.sqrt(2), -1/np.sqrt(2)),
#     '10' : complex(1/np.sqrt(2), -1/np.sqrt(2)),
# }
constellation_map = {
    0: 1,
    1: -1,
}



# Create known OFDM symbol
# Known OFDM symbol randomly chooses from the available constellation values

n_reps = 10			# Number of times to repeat the symbol

np.random.seed(0)	# Makes the random choices the same each time
source = np.random.choice(list(constellation_map.values()), (N-2)//2)
subcarrier_data = np.concatenate(([0], source, [0], np.conjugate(np.flip(source))))

x = np.fft.ifft(subcarrier_data).real
OFDM_symbol = np.concatenate((x[-L:], x))
OFDM_symbol /= np.max(np.abs(OFDM_symbol))

OFDM_frame = np.tile(OFDM_symbol, n_reps)



# Create transmitted signal
# Signal is 1 second silence, then 1 second chirp, then known OFDM symbol

t = np.linspace(0,1,fs)
chirp_standard = sig.chirp(t, 1e3, 1, 10e3, method='linear')
pause = np.zeros(fs)

signal = np.concatenate([pause, chirp_standard, OFDM_frame])
write('OFDM/test_frame.wav', fs, signal)



# Channel estimation

print('Recording...')
received = sd.rec(int(1.5*fs)+len(signal), samplerate = fs, channels = 1, blocking = True).flatten()
print('Recording finished')
write('OFDM/received_frame.wav', fs, received)
# received = np.array(read('OFDM/received_frame.wav', fs)[1])

chirp_end_index = synchronise(received, chirp_standard) + fs//2 - 1		# -1 fixes the synchronisation issue

received_frame = received[chirp_end_index:chirp_end_index + n_reps*(L+N)]

H_est = get_channel_estimate(subcarrier_data, received_frame, n_reps)
plot_channel(H_est)



# Decoding

r = received_frame[L:L+N]
R = np.fft.fft(r)

X_hat = R/H_est
plot_decoded_symbols(X_hat[:X_hat.shape[0]//2])