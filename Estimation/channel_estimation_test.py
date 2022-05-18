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
def get_chirp_end_index(sent_chirp, received_chirp):
	convolution = np.convolve(received_chirp, np.flip(sent_chirp))
	c = np.abs(convolution)

	signal_end_index = c.argmax()

	return signal_end_index

# Split an audio sample into a number of equilength sections
# Returns a dictionary where the key 'i' contains the i-th section of the sample
# MIGHT BE COMPUTATIONALLY FASTER IF YOU RETURN A BIG MATRIX WHERE THE i-TH ROW/COLUMN IS THE i-TH SECTION, CAN'T BE BOTHERED IMPLEMENTING THIS RN
def split_sample(data, section_length):
	section_dict = {}
	num_sections = 1 + data.size//section_length

	# Pad data with zeroes so it is a multiple of the section length
	padded_data = np.zeros(num_sections*section_length)
	padded_data[:data.size] = data
	
	split_data = np.split(padded_data, num_sections)
	
	for i in range(num_sections):
		section_dict[i] = split_data[i]

	return section_dict



# Transmitted and received test chirps for channel estimation
fs = 44100
chirp_f_start = 0
chirp_f_end = 20000
duration = 2
time_samples = np.arange(0, duration, 1/fs)

test_chirp = chirp(time_samples, chirp_f_start, duration, chirp_f_end, method = 'linear')
print('RECORDING')
received_chirp = sd.rec(int(2.5*duration*fs), samplerate = fs, channels = 1, blocking = True).flatten()
print('RECORDING FINISHED')

# Synchronise to find the start of the chirp in the received sample
signal_start_index = get_chirp_end_index(test_chirp, received_chirp) - duration*fs
section_length = 1024
num_sections = 1 + test_chirp.size//section_length

section_start_indicies = signal_start_index + np.linspace(0, section_length*num_sections, num_sections+1)

# Split the estimation chirps into bins by time section
test_chirp_sections = split_sample(test_chirp, section_length)
received_chirp_sections = split_sample(received_chirp[signal_start_index:], section_length)		# Truncation ensures only the received data from the chirp onwards is kept

assert len(received_chirp_sections.keys()) >= len(test_chirp_sections.keys())		# Test to ensure the received chirp has more sections than the transmitted chirp i.e. from the recording after the test chirp stops

# Plot the received signal and the indicies of the section it's divided into
#plt.plot(received_chirp, color = 'blue')
#plt.vlines(section_start_indicies, ymin = 0, ymax = 1, color = 'red')
#plt.show()


# Find the DFT of each section
X_est_sections = {}
for section_number in test_chirp_sections.keys():
	X_est_sections[section_number] = np.fft.fft(test_chirp_sections[section_number])/np.sqrt(section_length)

Y_est_sections = {}
for section_number in received_chirp_sections.keys():
	Y_est_sections[section_number] = np.fft.fft(received_chirp_sections[section_number])/np.sqrt(section_length)

# Find the corresponding start frequency of the section
# Currently using a linear 0Hz-20kHz chirp to estimate the channel so the sample index (time) is proportional to the freqeuncy
bin_freq_range = (chirp_f_end-chirp_f_start)/num_sections

# Find the channel FFT in each frequency bin by dividing corresponding sections in X and Y
H_estimated_sections = {}
for section_number in test_chirp_sections.keys():
	X = X_est_sections[section_number]
	Y = Y_est_sections[section_number]

	bin_start_freq = chirp_f_start + section_number*bin_freq_range

	try:
		H_estimated_sections[bin_start_freq] = Y/X
	except:			# In case one of the sections in X is full of zeroes, giving a divide by zero error
		H_estimated_sections[bin_start_freq] = np.zeros(section_length)

H_full = np.concatenate(list(H_estimated_sections.values()))
freq_samples = np.linspace(chirp_f_start, chirp_f_end, num_sections*section_length)

plt.plot(freq_samples, np.abs(H_full))
plt.xlabel('Frequency (Hz)')
plt.ylabel('|H(f)|')
plt.show()