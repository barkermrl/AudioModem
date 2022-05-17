import numpy as np
from scipy.io.wavfile import write
from scipy.signal import chirp 
import sounddevice as sd
import matplotlib.pyplot as plt

def gen_chirp(f0, f1, T, save = False, play = True, form = 'logarithmic', repititions = 1):
    fs = 44100
    t = np.arange(0, int(T*fs))/fs
    w = chirp(t, f0, T, f1, method = form)
    if repititions > 1:
        w = np.tile(w, repititions)
    if save == True:
        write('chirp_{}_{}_{}.wav'.format(form, f0, f1), fs, w)
    if play == True:
        sd.play(w, fs, blocking = True)

def record(title, duration = 5, save = False, plot = False):
    fs = 44100
    r = sd.rec(int(duration*fs), samplerate = fs, channels = 1, blocking = True).flatten()
    if save  == True:
        write(title, fs, r)
    if plot == True:
        plt.plot(np.arange(0, int(duration*fs)), r)
        plt.show()
    return r
    