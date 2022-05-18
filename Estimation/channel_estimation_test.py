"""
Test for channel estimation by comparing a known transmitted chirp to its recieved counterpart
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import struct, wave
import soundfile as sf
from utilities import *



# Converts a 32 bit floating point .wav files into int16 PCM .wav files for compatability with the wave module
# 'filename' given WITH the .wav file type extension
def float32_to_int16(filename):
	# Extract the 32 bit floating point data fand samplerate rom the given .wav file
	data_32bfp, samplerate = sf.read(filename+'.wav', dtype='float32')
	data_32bfp = np.asarray(data_32bfp)

	int16_dtype = np.dtype('int16')
	
	# Convert the 32 bit floating point values in the file into int16 values
	i = np.iinfo(int16_dtype)
	abs_max = 2 ** (i.bits - 1)
	offset = i.min + abs_max
	data_int16 = (data_32bfp * abs_max + offset).clip(i.min, i.max).astype(int16_dtype)

	# Write the int16 .wav file
	saved_filename = filename+'_int16.wav'
	sf.write(saved_filename, data_int16, samplerate)

# Returns a numpy array containing the samples from a given .wav file
# 'filename' given WITH the .wav file type extension
def samples_from_wav(filename):
	w = wave.open(filename)
	frames = w.readframes(w.getnframes())
	samples = np.asarray([x[0] for x in struct.iter_unpack('<h', frames)])

	return samples

# Finds the start index of the received chirp by convolving with the time-reversed known chirp
# Returns the start index of the chirp in the unknown signal array
def get_chirp_start_index(known_signal, unknown_signal):
	convolution = np.convolve(y_test, np.flip(x_test))
	signal_start_index = np.argmax(abs(convolution))

	return signal_start_index

# Split an audio sample into a number of equilength sections
# Returns a dictionary where the key 'i' contains the i-th section of the sample
# WILL PROBABLY END UP BEING COMPUTATIONALLY FASTER IF YOU RETURN A BIG MATRIX WHERE THE i-TH ROW/COLUMN IS THE i-TH SECTION, CBA IMPLEMENTING THIS RN
def split_sample(data, section_length):
	section_dict = {}
	num_sections = data.size//section_length

	# Pad data with zeroes so it is a multiple of the section length
	padded_data = np.pad(data, num_sections*section_length - data.size)
	split_data = np.split(padded_data, num_sections)
	
	for i in range(num_sections):
		section_dict[i] = split_data[i]

	return section_dict



# Transmitted and received test chirps
transmitted_estimator_filename = 'channel_test_transmitted_int16.wav'
received_estimator_filename = 'channel_test_received_int16.wav'

x_est = samples_from_wav(transmitted_estimator_filename)
y_est = samples_from_wav(received_estimator_filename)

# Synchronise to find the start of the chirp in the received sample
signal_start_index = get_chirp_start_index(x_test, y_test)
num = 100
section_start_indicies = signal_start_index + np.linspace(0, section_length*num, num+1)

# Split the estimation chirps into bins by time section
x_est_sections = split_sample(x_est, section_length)
y_est_sections = split_sample(y_est[signal_start_index:], section_length)

# Plot the received signal and the indicies of the section it's divided into
plt.plot(np.abs(y_test), color = 'blue')
plt.vlines(section_start_indicies+, color = 'red')
plt.show()

"""
print('x length = {}\ny length = {}\nconv length = {}\nsignal start index = {}'.format(x.size, y.size, convolution.size, signal_start_index))
plt.plot(convolution)
plt.show()



# Calculate FFTs of the received and transmitted chirps. The received signal y is truncated to the start of the chirp being detected at the peak of the convolution
y_trunc = y[signal_start_index:]
Y = np.fft.fft(y_trunc)
Y /= np.sqrt(Y.size)

X = np.fft.fft(x, n=y_trunc.size)
X /= np.sqrt(X.size)


# Least-Squares channel frequency estimate
H_ls = X/Y

fs = 44100
fft_length = H_ls.size
freq_samples = np.linspace(0, fs, fft_length)
H_ls_max = np.amax(np.abs(H_ls))

# Dict contains key, value pairs as follows:
# Key =  is the noise noise cut-off point i.e. noisy frequency components are decided to be that percentage of the max absolute value of H_ls
# Value is the resulting array, removing all noisy components form H_ls
H_ls_nonoise_dict = {
	0.2 : [],
	0.1 : [],
	0.05 : [],
	0.01 : [],
}

for cutoff in H_ls_nonoise_dict.keys():
	mask = np.abs(H_ls) > cutoff * H_ls_max
	H_ls_nonoise_dict[cutoff] = H_ls[mask]


# Plot the channel frequency spectrum for positive frequencies; the channel is real so the DFT is symmetric
# Assume all frequency components with values < 0.1*max{H_ls} are negligible
plt.plot(freq_samples, np.abs(H_ls_nonoise_dict[0.1]))
plt.show()
"""