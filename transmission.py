# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
import sounddevice as sd
import subprocess
import tempfile


class Chirp:
    def __init__(self, fmin=1e3, fmax=10e3, duration=1, fs=48000):
        self.fmin = fmin
        self.fmax = fmax
        self.duration = duration
        self.fs = fs

        # Create chirp signal using scipy chirp function.
        t = np.linspace(0, self.duration, self.fs)
        self.signal = sig.chirp(t, fmin, duration, fmax, method="linear")


class Transmission:
    def __init__(self, source, constellation_map, L=1024, N=8192, fs=48000):
        self.source = source
        self.L = L
        self.N = N
        self.fs = fs
        self.constellation_map = constellation_map

        # Calculate number of carriers used to transmit information
        self.num_data_carriers = (self.N - 2) // 2

        # Create signal
        symbols = self._create_symbols()
        header = self._create_header()
        body = self._create_body(symbols)
        self.signal = np.concatenate((header, body))

    def _create_symbols(self):
        symbols = []

        for source_chunk in self._chunk_source():
            subcarrier_data = np.concatenate(
                ([0], source_chunk, [0], np.conjugate(np.flip(source)))
            )

            x = np.fft.ifft(subcarrier_data).real
            OFDM_symbol = np.concatenate((x[-self.L :], x))
            OFDM_symbol /= np.max(np.abs(OFDM_symbol))

            symbols.append(OFDM_symbol)
        return symbols

    def _chunk_source(self):
        # Need to pad the source with end zeros if required.
        # Calculate remainder when length of source is divided by number of
        # carriers used to transmit information.
        end_zeros = np.zeros(
            self.num_data_carriers - (len(self.source) % self.num_data_carriers)
        )
        source_with_end_zeros = np.concatenate([self.source, end_zeros])

        # Reshape 1D array into 2D array with shape  (Y x num_data_carriers).
        shape = (
            len(source_with_end_zeros) // self.num_data_carriers,
            self.num_data_carriers,
        )
        return source_with_end_zeros.reshape(shape)

    def _create_header(self):
        pause = np.zeros(self.fs)
        self.chirp = Chirp(fmin=1e3, fmax=10e3, duration=1, fs=self.fs)
        return np.concatenate([pause, self.chirp.signal, pause])

    def _create_body(self, symbols):
        # TODO Include periodic resynchronisations/estimations etc.
        # For now only concatenates the symbols into one long signal.
        return np.concatenate(symbols)

    def save_signals(self, t_fname="files/signal.wav", r_fname="files/received.wav"):
        wav.write(r_fname, self.fs, self.received_signal)
        wav.write(t_fname, self.fs, self.signal)

    def load_signals(self, t_fname="files/signal.wav", r_fname="files/received.wav"):
        # Read transmitted signal to ensure it matches signal tranmission is
        # expecting.
        fs, signal = wav.read(t_fname)
        assert self.signal == signal
        assert self.fs == fs

        self.received_signal = wav.read(r_fname)

    def record_signal(self, afplay=False):
        if afplay:
            # Requires afplay to be installed as a terminal command!
            wav.write("tmp.wav", self.fs, self.signal)
            subprocess.Popen(["afplay", "tmp.wav"])

        print("Recording...")
        self.received_signal = sd.rec(
            int(1.5 * self.fs) + len(self.signal),
            samplerate=self.fs,
            channels=1,
            blocking=True,
        ).flatten()
        print("Recording finished")


# Define functions

# Returns the middle index of a synchronisation chirp
def synchronise(received_signal, transmitted_chirp):
    def window(x):
        return x * np.hanning(len(x))

    c = sig.convolve(
        window(received_signal), np.flip(window(transmitted_chirp)), mode="same"
    )

    return np.argmax(np.abs(c))


# Returns a channel estimate from the known and received OFDM symbols
def get_channel_estimate(known_symbol, received_symbols, num_symbols):

    R = np.zeros((num_symbols, known_symbol.shape[0]), dtype=complex)

    for i in range(num_symbols):
        r = received_symbols[L + i * (N + L) : (i + 1) * (L + N)]
        R[i, :] = np.fft.fft(r)

    H_est = np.mean(R, axis=0) / known_symbol

    return H_est


# Plots a channel's frequency response and phase
def plot_channel(H, fs=48000):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5))

    freqs = np.linspace(0, fs, H.shape[0])

    ax_left.plot(freqs, 10 * np.log10(np.abs(H)))
    ax_left.set_xlabel("Frequency [Hz]")
    ax_left.set_ylabel("Magnitude [dB]")

    ax_right.scatter(freqs, np.angle(H))
    ax_right.set_xlabel("Frequency [Hz]")
    ax_right.set_ylabel("Phase [rad]")
    ax_right.set_ylim(-np.pi, np.pi)

    plt.show()


# Plots decoded symbols coloured by frequency bin
def plot_decoded_symbols(
    X,
):
    plt.scatter(X.real, X.imag, c=np.arange(len(X)), cmap="gist_rainbow_r", marker=".")

    plt.title("Decoded constellation symbols")
    plt.xlabel("Re")
    plt.ylabel("Im")
    cbar = plt.colorbar()
    cbar.set_label("Subcarrier number")

    plt.show()


# Create known OFDM symbol
# Known OFDM symbol randomly chooses from the available constellation values

np.random.seed(0)  # Makes the random choices the same each time
constellation_map = {
    "00": complex(1 / np.sqrt(2), 1 / np.sqrt(2)),
    "01": complex(-1 / np.sqrt(2), 1 / np.sqrt(2)),
    "11": complex(-1 / np.sqrt(2), -1 / np.sqrt(2)),
    "10": complex(1 / np.sqrt(2), -1 / np.sqrt(2)),
}
#         self.constellation_map = {
#             0: 1,
#             1: -1,
#         }

known_symbol = np.random.choice(list(constellation_map.values()), (8192 - 2) // 2)
source = np.tile(known_symbol, 5)

transmission = Transmission(source, constellation_map)
transmission.record_signal(afplay=True)
transmission.save_signals()

chirp_end_index = (
    synchronise(received, chirp.signal) + fs // 2
)  # -1 fixes the synchronisation issue

received_frame = received[chirp_end_index : chirp_end_index + n_reps * (L + N)]

H_est = get_channel_estimate(subcarrier_data, received_frame, n_reps)
plot_channel(H_est)


# Decoding

r = received_frame[L : L + N]
R = np.fft.fft(r)

X_hat = R / H_est
# plot_decoded_symbols(X_hat[:X_hat.shape[0]//2])
plot_decoded_symbols(X_hat[1000:1500])
