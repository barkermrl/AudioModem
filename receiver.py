# Import libraries
from transmission import Transmission

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
from scipy.optimize import minimize
import sounddevice as sd
import subprocess
import LDPC_utils
from bitarray import bitarray

# Only use frequency bins 86-853 inclusive
c = LDPC_utils.get_code()

FREQ_MIN = 86
FREQ_MAX = 854

N_BINS = FREQ_MAX - FREQ_MIN


CONSTELLATION_MAP = {
    "00": complex(1 / np.sqrt(2), 1 / np.sqrt(2)),
    "01": complex(-1 / np.sqrt(2), 1 / np.sqrt(2)),
    "11": complex(-1 / np.sqrt(2), -1 / np.sqrt(2)),
    "10": complex(1 / np.sqrt(2), -1 / np.sqrt(2)),
}

KEYS = list(CONSTELLATION_MAP.keys())
VALUES = list(CONSTELLATION_MAP.values())


FS = 48000
L = 512
N = 4096
FRAME_LENGTH = 128
NUM_FRAME_VALUES = FRAME_LENGTH * (L + N)

NUM_CARRIERS = (N - 2) // 2
KNOWN_SYMBOL_BIG_X = np.load("known_ofdm_symbol.npy")  # complex constellation values
KNOWN_SYMBOL_SMALL_X = np.fft.ifft(KNOWN_SYMBOL_BIG_X).real
KNOWN_SYMBOL_SMALL_X /= np.max(np.abs(KNOWN_SYMBOL_SMALL_X))
GUARD = KNOWN_SYMBOL_SMALL_X[-L:]


t = np.linspace(0, 1, FS)
CHIRP = sig.chirp(t, 1e3, 1, 10e3, method="linear")
PREAMBLE = np.concatenate(
    [
        CHIRP,
        GUARD,
        KNOWN_SYMBOL_SMALL_X,
        KNOWN_SYMBOL_SMALL_X,
        KNOWN_SYMBOL_SMALL_X,
        KNOWN_SYMBOL_SMALL_X,
    ]
)
ENDAMBLE = np.concatenate(
    [
        GUARD,
        KNOWN_SYMBOL_SMALL_X,
        KNOWN_SYMBOL_SMALL_X,
        KNOWN_SYMBOL_SMALL_X,
        KNOWN_SYMBOL_SMALL_X,
        CHIRP,
    ]
)

transmission = Transmission(np.zeros(N_BINS))
# transmission.record_signal()
# transmission.save_signals(t_fname="files/tmp.wav")
#transmission.load_signals(r_fname="files/frenzy_test.wav")
transmission.received_signal = np.load('Group4_rec.npy')
#transmission.received_signal = np.load('Group9_rec.npy')
# Put Xhats, vars from each for loop iteration in here
all_constellation_vals = []
all_vars = []

for ind_frame, frame in enumerate(transmission.get_frames()):
    frame = frame/np.max(frame)
    plt.title(f'Frame {ind_frame}')
    plt.plot(frame)
    plt.show()
    transmission.received_signal = frame
    transmission.synchronise()
    transmission.estimate_H()
    transmission.Xhats_estimate()

    for block_number, block_xhats in enumerate(transmission.Xhats):
        all_constellation_vals += block_xhats
        #plt.title(f'{block_number}')
        #plt.xlim(-5,5)
        #plt.ylim(-5,5)
        #plt.scatter(np.array(block_xhats).real, np.array(block_xhats).imag, s = 1, c = np.arange(len(block_xhats)))
        #plt.show()
        
    plt.title(f'Received Symbols')
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.scatter(np.array(all_constellation_vals).real, np.array(all_constellation_vals).imag, s = 1)
    plt.show()

    all_vars += transmission.vars.tolist()

# print(len(all_constellation_vals), all_constellation_vals)

# Throw away last (len(all_constellation_vals) % c.K) constellation values, they are only used for padding
remainder = len(all_constellation_vals) % c.K
print(remainder)
if remainder != 0:
    constellation_vals_no_padding = all_constellation_vals[
        : len(all_constellation_vals) - remainder
    ]
    all_vars_no_padding = all_vars[: len(all_constellation_vals) - remainder]
else:
    constellation_vals_no_padding = all_constellation_vals
    all_vars_no_padding = all_vars

# Split arrays into chunks of length c.K
print(len(constellation_vals_no_padding), len(all_vars_no_padding))
constellation_vals_no_padding = np.reshape(constellation_vals_no_padding, (-1, c.K))
all_vars_no_padding = np.reshape(all_vars_no_padding, (-1, c.K))

bit_array = []
for codeword, var in zip(constellation_vals_no_padding, all_vars_no_padding):

    # Find LLRs from codewords
    llrs = np.array(LDPC_utils.get_llr(codeword, var), dtype=np.float64)

    # Put LLRs into decoder, get out binary file
    u_hats = LDPC_utils.decode(llrs, c)
    bit_array += u_hats

data_bits = bitarray(endian="little")
data_bits.extend(bit_array)
# Turn bit array into bytes
data_bytes = data_bits.tobytes()
print(data_bytes[:100])

# Extract header from byte array
null_terminator_index = 0
for b in data_bytes:
    null_terminator_index += 1
    if b == "\0":
        break


header = data_bytes[:null_terminator_index]

filelength_asbytes = header[:4]
filelength_asint = int.from_bytes(filelength_asbytes, byteorder="little")

filename = header[4:]

# Turn into file
filedata = data_bytes[
    1 + null_terminator_index : 1 + null_terminator_index + filelength_asint
]
# write(f"{filename}", filedata)
