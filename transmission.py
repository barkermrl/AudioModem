# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
import sounddevice as sd
import subprocess


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
    def __init__(self, source, constellation_map, L=512, N=4096, fs=48000):
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
        footer = self._create_footer()
        self.signal = np.concatenate((header, body, footer))

    def _create_symbols(self):
        symbols = []
        self.Xs = []

        for source_chunk in self._chunk_source():
            subcarrier_data = np.concatenate(
                ([0], source_chunk, [0], np.conjugate(np.flip(source_chunk)))
            )
            self.Xs.append(subcarrier_data)
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
        return np.concatenate([pause, self.chirp.signal])

    def _create_footer(self):
        self.chirp = Chirp(fmin=1e3, fmax=10e3, duration=1, fs=self.fs)
        pause = np.zeros(self.fs)
        return np.concatenate([self.chirp.signal, pause])

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
        np.testing.assert_equal(self.signal, signal)
        assert self.fs == fs

        fs, self.received_signal = wav.read(r_fname)

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

    def synchronise(self, n):
        # End of chirp is half a second after chirp.
        # Frame starts 1 second later.
        peaks = self._find_chirp_peaks()
        frame_start_index = peaks[0] + self.fs // 2
        frame_end_index = peaks[-1] - self.fs // 2
        print(frame_start_index, frame_end_index)
        self.Rs = self._identify_Rs(frame_start_index, n)

    def _find_chirp_peaks(self):
        # window = lambda x: x * np.hanning(len(x))
        conv = sig.convolve(
            self.received_signal,
            np.flip(self.chirp.signal),
            mode="same",
        )
        peaks = sig.find_peaks(
            conv, distance=transmission.fs, height=200, prominence=3, threshold=3
        )[0]
        return peaks

    def _identify_Rs(self, frame_start_index, n):
        Rs = []
        for i in range(n):
            start = frame_start_index + (self.L + self.N) * i + self.L
            end = frame_start_index + (self.L + self.N) * (i + 1)
            r = self.received_signal[start:end]
            R = np.fft.fft(r)
            Rs.append(R)
        return Rs

    def estimate_H(self, n):
        # Returns a channel estimate from the known and received OFDM symbols
        R = np.vstack(self.Rs)
        self.H_est = np.mean(R, axis=0) / self.Xs[0]

    def estimate_Xhats(self):
        self.Xhats = []

        for R in self.Rs:
            X = R / self.H_est

            # Only add useful part of carrier data X
            # Bins determined by standard (86 to 854 inclusive)
            self.Xhats.append(X[86:854])

    def plot_channel(self):
        _, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5))

        freqs = np.linspace(0, self.fs, self.H_est.shape[0])

        ax_left.plot(freqs, 10 * np.log10(np.abs(self.H_est)))
        ax_left.set_xlabel("Frequency [Hz]")
        ax_left.set_ylabel("Magnitude [dB]")

        ax_right.scatter(freqs, np.angle(self.H_est))
        ax_right.set_xlabel("Frequency [Hz]")
        ax_right.set_ylabel("Phase [rad]")
        ax_right.set_ylim(-np.pi, np.pi)

        plt.show()

    def plot_decoded_symbols(self):
        # Plots decoded symbols coloured by frequency bin
        X = self.Xhats[-1]
        plt.scatter(
            X.real, X.imag, c=np.arange(len(X)), cmap="gist_rainbow_r", marker="."
        )

        plt.title("Decoded constellation symbols")
        plt.xlabel("Re")
        plt.ylabel("Im")
        cbar = plt.colorbar()
        cbar.set_label("Subcarrier number")

        plt.show()


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

# Known OFDM symbol randomly chooses from the available constellation values
known_symbol = np.random.choice(list(constellation_map.values()), (4096 - 2) // 2)
n = 50
source = np.tile(known_symbol, n)

fs = 48000

transmission = Transmission(source, constellation_map, fs=fs)
# transmission.record_signal(afplay=True)
# transmission.save_signals()
transmission.load_signals()

transmission.synchronise(n)
transmission.estimate_H(n)
transmission.estimate_Xhats()

transmission.plot_channel()
transmission.plot_decoded_symbols()

breakpoint()
