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
	data_32bfp, samplerate = sf.read(filename, dtype='float32')
	data_32bfp = np.asarray(data_32bfp)

	int16_dtype = np.dtype('int16')
	
	# Convert the 32 bit floating point values in the file into int16 values
	i = np.iinfo(int16_dtype)
	abs_max = 2 ** (i.bits - 1)
	offset = i.min + abs_max
	data_int16 = (data_32bfp * abs_max + offset).clip(i.min, i.max).astype(int16_dtype)

	# Write the int16 .wav file
	saved_filename = filename[:-4]+'_int16.wav'
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
# MIGHT BE COMPUTATIONALLY FASTER IF YOU RETURN A BIG MATRIX WHERE THE i-TH ROW/COLUMN IS THE i-TH SECTION, CAN'T BE BOTHERED IMPLEMENTING THIS RN
def split_sample(data, section_length):
	section_dict = {}
	num_sections = 1 + data.size//section_length

	# Pad data with zeroes so it is a multiple of the section length
	padded_data = np.pad(data, num_sections*section_length - data.size)
	split_data = np.split(padded_data, num_sections)
	
	for i in range(num_sections):
		section_dict[i] = split_data[i]

	return section_dict



# Transmitted and received test chirps for channel estimation
transmitted_estimator_filename = 'channel_test_transmitted_int16.wav'
received_estimator_filename = 'channel_test_received_int16.wav'

x_est = samples_from_wav(transmitted_estimator_filename)
y_est = samples_from_wav(received_estimator_filename)

# Synchronise to find the start of the chirp in the received sample
signal_start_index = get_chirp_start_index(x_test, y_test)
section_length = 1024
num = 100
section_start_indicies = signal_start_index + np.linspace(0, section_length*num, num+1)

# Split the estimation chirps into bins by time section
x_est_sections = split_sample(x_est, section_length)
y_est_sections = split_sample(y_est[signal_start_index:], section_length)		# Truncation ensures only the received data from the chirp onwards is kept

assert len(y_est_sections.keys()) >= len(x_est_sections.keys())		# Test to ensure the received chirp has more sections than the transmitted chirp i.e. from the recording after the test chirp stops

# Plot the received signal and the indicies of the section it's divided into
plt.plot(np.abs(y_test), color = 'blue')
plt.vlines(section_start_indicies, color = 'red')
plt.show()


# Find the DFT of each section
X_est_sections = {}
for section_number in x_est_sections.keys():
	X_est_sections[section_number] = np.fft.fft[x_est_sections[section_number]]/np.sqrt(section_length)

Y_est_sections = {}
for section_number in y_est_sections.keys():
	Y_est_sections[section_number] = np.fft.fft[y_est_sections[section_number]]/np.sqrt(section_length)

# Find the corresponding start frequency of the section
# Currently using a linear 0Hz-20kHz chirp to estimate the channel so the sample index (time) is proportional to the freqeuncy
chirp_start_freq = 0
chirp_end_freq = 20000
num_bins = 1 + x_est.size//section_length

bin_freq_range = (chirp_end_freq-chirp_start_freq)/num_bins

# Find the channel FFT in each frequency bin by dividing corresponding sections in X and Y
H_estimated_sections = {}
for section_number in x_est_sections.keys():
	X = X_est_sections[section_number]
	Y = Y_est_sections[section_number]

	bin_start_freq = chirp_start_freq + section_number*bin_freq_range

	try:
		H_estimated_sections[bin_start_freq] = Y/X
	except:			# In case one of the sections in X is full of zeroes, giving a divide by zero error
		H_estimated_sections[bin_start_freq] = np.zeros(section_length)


full_H = np.concatenate((H_estimated_sections.values()))
plt.plot(np.abs(full_H))
plt.vlines(H_estimated_sections.keys())

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