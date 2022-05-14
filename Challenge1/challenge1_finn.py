# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from bitarray import bitarray

# Get the channel impulse response and its FFT
channel = np.genfromtxt('channel.csv')
H = np.fft.fft(channel, n=1024)

# Get the received file; each OFDM symbol is a length 1024 IDFT with a 32 bit cyclic prefix
#file_name = str(input('Enter file name (format is fileX.csv): '))
received = np.genfromtxt('file1.csv')
#print('File has {} lines'.format(received.size))

# Get the DFT of the data blocks from a received file
cp_len = 32
data_len = 1024
symbol_len = cp_len + data_len
num_OFDM_symbols = received.size//symbol_len
	
# The n-th key of this dict contains the n-th block of DFT data
DFT_dict = {}
	
for i in range(num_OFDM_symbols):

	block_start_index = cp_len + symbol_len*i
	block_end_index = symbol_len + symbol_len*i

	IDFT_block = received[block_start_index:block_end_index]
	DFT_block = np.fft.fft(IDFT_block)

	DFT_dict[i] = DFT_block

# Find the received encoded values and select the unique information elements
X_dict = {}
for key in DFT_dict:

	X = DFT_dict[key]/H
	X /= np.sqrt(1024)		# Normalise to correct for the size of X

	X_dict[key] = X[1:512]

"""
# Plot the received values of the first block to make sure it's decoded correctly
# Should see a noisy QPSK constellation
plt.scatter(np.real(X_dict[0]), np.imag(X_dict[0]))
plt.show()
"""

# Set up the QPSK constellation map that encoded the bits
constellation_map = {
    '00' : complex(1/np.sqrt(2), 1/np.sqrt(2)),
    '01' : complex(-1/np.sqrt(2), 1/np.sqrt(2)),
    '11' : complex(-1/np.sqrt(2), -1/np.sqrt(2)),
    '10' : complex(1/np.sqrt(2), -1/np.sqrt(2)),
}

QPSK_symbols = np.array([*constellation_map.values()])
QPSK_bits = np.array([*constellation_map.keys()])

# Decode the bits, then turn the bytes into ASCII characters
byte = ''
letters = ''
count = 0

for key in X_dict:
	for symbol in X_dict[key]:	
		byte += QPSK_bits[(np.abs(QPSK_symbols - symbol)).argmin()]
	
		count += 1
		if count == 4:
			"""
			# chr() returns the unicode letter associated with the given decimal integer
			if byte == '00000000':
				letters += 'NUL'
			else:
				letters += chr(int(byte, 2))
			"""
			letters += bitarray(byte).tobytes().decode('ascii', errors='ignore')
			count = 0
			byte = ''

print(letters)