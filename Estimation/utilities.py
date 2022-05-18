import numpy as np
from scipy.io.wavfile import write, read
from scipy.signal import chirp 
import sounddevice as sd
import matplotlib.pyplot as plt

def gen_chirp(T, save = False, play = False, form = 'logarithmic', repititions = 1):
    fs = 44100
    t = np.arange(0, int(T*fs))/fs
    w = chirp(t, 20, T, 20000, method = form)
    if repititions > 1:
        w = np.tile(w, repititions)
    if save == True:
        write('chirp_{}_{}_{}.wav'.format(form, 20, 20000), fs, w)
    if play == True:
        sd.play(w, fs, blocking = True)
    return w

def record(title = 'test', duration = 5, save = False, plot = False):
    fs = 44100
    r = sd.rec(int(duration*fs), samplerate = fs, channels = 1, blocking = True).flatten()
    if save  == True:
        write("{}.wav".format(title), fs, r)
    if plot == True:
        plt.plot(np.arange(0, int(duration*fs)), r)
        plt.show()
    return r

def conv(r, w):
    fs = 44100
    w_inv = np.flip(w)
    convolution = np.convolve(r,w_inv)
    t = np.arange(0, len(convolution))/fs
    plt.plot(t, np.abs(convolution))
    plt.show()
    
def read_wav(filename):
    fs, data = read(filename)
    return data

