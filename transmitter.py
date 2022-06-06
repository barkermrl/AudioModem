# Import libraries
from transmission import Transmission

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
from scipy.optimize import minimize
import sounddevice as sd
import subprocess

# Only use frequency bins 86-853 inclusive
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

START = np.concatenate([np.zeros(FS), CHIRP])
END = np.concatenate([CHIRP, np.zeros(FS)])




source = np.load("frenzy_constellation_values.npy")

num_extra_values = len(source) % N_BINS

np.random.seed(0)
source = np.append(source, np.random.choice(VALUES, num_extra_values))

np.seterr(all="ignore")  # Supresses runtime warnings

transmission = Transmission(source)
transmission.play_signal()