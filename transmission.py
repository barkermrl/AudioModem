# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav
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

NUM_CARRIERS = (N - 2) // 2
KNOWN_SYMBOL = np.load("known_ofdm_symbol.npy")
GUARD = KNOWN_SYMBOL[-L:]


t = np.linspace(0, 1, FS)
CHIRP = sig.chirp(t, 1e3, 1, 10e3, method="linear")
PREAMBLE = np.concatenate(
    [CHIRP, GUARD, KNOWN_SYMBOL, KNOWN_SYMBOL, KNOWN_SYMBOL, KNOWN_SYMBOL]
)
ENDAMBLE = np.concatenate(
    [GUARD, KNOWN_SYMBOL, KNOWN_SYMBOL, KNOWN_SYMBOL, KNOWN_SYMBOL, CHIRP]
)

START = np.concatenate([np.zeros(FS), CHIRP])
END = np.concatenate([CHIRP, np.zeros(FS)])


class Transmission:
    def __init__(self, source):
        self.source = source
        self.source_chunks = source.reshape((-1, N_BINS))

        # Create signal
        symbols = self._create_symbols()
        frame = np.concatenate([PREAMBLE, symbols, ENDAMBLE])

        # TODO: Add file size in header
        self.signal = np.concatenate((START, frame, END))

    def _create_symbols(self):
        symbols = []
        self.Xs = []

        for source_chunk in self.source_chunks:
            data_carriers = np.concatenate(
                (
                    np.random.choice(VALUES, NUM_CARRIERS - FREQ_MIN),
                    source_chunk,
                    np.random.choice(VALUES, NUM_CARRIERS - FREQ_MAX),
                )
            )
            subcarrier_data = np.concatenate(
                ([0], data_carriers, [0], np.conjugate(np.flip(data_carriers)))
            )
            self.Xs.append(subcarrier_data)
            x = np.fft.ifft(subcarrier_data).real
            OFDM_symbol = np.concatenate((x[-L:], x))
            OFDM_symbol /= np.max(np.abs(OFDM_symbol))

            symbols.append(OFDM_symbol)
        return np.concatenate(symbols)

    def save_signals(self, t_fname="files/signal.wav", r_fname="files/received.wav"):
        wav.write(r_fname, FS, self.received_signal)
        wav.write(t_fname, FS, self.signal)

    def load_signals(self, t_fname="files/signal.wav", r_fname="files/received.wav"):
        # Read transmitted signal to ensure it matches signal tranmission is
        # expecting.
        fs, signal = wav.read(t_fname)
        np.testing.assert_equal(self.signal, signal)
        assert fs == FS

        fs, self.received_signal = wav.read(r_fname)
        assert fs == FS

    def record_signal(self, afplay=False):
        if afplay:
            # Requires afplay to be installed as a terminal command!
            wav.write("tmp.wav", FS, self.signal)
            subprocess.Popen(["afplay", "tmp.wav"])

        print("Recording...")
        self.received_signal = sd.rec(
            int(1.5 * FS) + len(self.signal),
            samplerate=FS,
            channels=1,
            blocking=True,
        ).flatten()
        print("Recording finished")

    def synchronise(self, offset=5, plot=False):
        # End of chirp is half a second after chirp.
        # Offset gives the number of samples to shift back by to ensure you're sampling early.
        peaks = np.sort(self._find_chirp_peaks())

        # peaks should be length 4 (1 chirp at the start and end of signal)
        # In addition, each frame starts and ends with a chirp.
        frame_start_index = peaks[1] + len(CHIRP) // 2 + len(PREAMBLE) - offset
        frame_end_index = peaks[2] - len(CHIRP) // 2 + len(PREAMBLE) - offset

        num_symbols = np.round((frame_end_index - frame_start_index) / (L + N))

        sampling_error = frame_end_index - frame_start_index - num_symbols * (L + N)

        print(
            "Peaks: {}\nFrame start: {}\nFrame end: {}\nNum symbols: {}\nSampling error: {}".format(
                peaks, frame_start_index, frame_end_index, num_symbols, sampling_error
            )
        )

        self.Rs = self._identify_Rs(frame_start_index, num_symbols, sampling_error)

        self.known_symbols_start = self.received_signal[
            frame_start_index - 4 * N : frame_start_index
        ]
        self.known_symbols_end = self.received_signal[
            frame_end_index + L : frame_end_index + L + 4 * N
        ]

        if plot:
            plt.plot(self.received_signal)
            plt.axvline(
                frame_start_index, color="r", linestyle=":", label="Frame start/end"
            )
            plt.axvline(frame_end_index, color="r", linestyle=":")

            plt.title("Received signal")
            plt.xlabel("Sample index")
            plt.ylabel("Amplitude")
            plt.legend(loc="upper right")

            plt.show()

    def _find_chirp_peaks(self):
        # window = lambda x: x * np.hanning(len(x))
        conv = sig.convolve(
            self.received_signal,
            np.flip(CHIRP),
            mode="same",
        )
        peaks = sig.find_peaks(conv, height=0.5 * np.abs(conv).max(), distance=FS - 1)[
            0
        ]
        return peaks

    def _identify_Rs(self, frame_start_index, num_symbols, sampling_error):
        Rs = []
        error_per_symbols = sampling_error / num_symbols

        for i in range(n):
            start = int(
                frame_start_index + L + np.rint((L + N + error_per_symbols) * i)
            )
            end = start + N
            r = self.received_signal[start:end]
            R = np.fft.fft(r)
            Rs.append(R)
        return Rs

    def estimate_H(self):
        # Returns a channel estimate from the known and received OFDM symbols
        self.H_est = self._complex_average(self.known_symbols_start)
        self.H_est_end = self._complex_average(self.known_symbols_end)

    def _complex_average(self, known_symbols):
        R = known_symbols.reshape((-1, N))
        magnitudes = np.mean(np.abs(R / KNOWN_SYMBOL), axis=0)
        angles = np.mean(np.angle(R / KNOWN_SYMBOL), axis=0)
        # np.mean(R / X, axis=0)
        return magnitudes * np.exp(1j * angles)

    def estimate_Xhats(self):
        self.Xhats = []

        for R in self.Rs:
            X = R / self.H_est

            # Only add useful part of carrier data X
            # Bins determined by standard (86 to 854 inclusive)
            self.Xhats.append(X[FREQ_MIN:FREQ_MAX])

    def _unwrap_phases(self):
        # Correct for wrapping of phases to [pi, -pi]
        phases = np.angle(self.H_est)
        for i in range(phases.shape[0] - 1):
            diff = phases[i + 1] - phases[i]
            if diff >= np.pi:
                phases[i + 1 :] -= 2 * np.pi
            elif diff <= -np.pi:
                phases[i + 1 :] += 2 * np.pi

        return phases

    def _find_drift(self, plot=False):
        # Correct for wrapping of phases to [pi, -pi]
        phases = self._unwrap_phases()

        # Correct for linear trend from incorrect syncronisation
        phase_linear_trend = np.linspace(0, phases[-1], phases.shape[0])
        corrected_phases = phases - phase_linear_trend

        if plot:
            freqs = np.arange(self.H_est.shape[0])

            plt.scatter(
                freqs,
                np.angle(self.H_est),
                color="blue",
                marker=".",
                label="Wrapped phases",
            )
            plt.scatter(
                freqs, phases, color="green", marker=".", label="Unwrapped phases"
            )
            plt.scatter(
                freqs,
                phase_linear_trend,
                color="grey",
                marker=".",
                label="Linear trend",
            )
            plt.scatter(
                freqs,
                corrected_phases,
                color="red",
                marker=".",
                label="Corrected phases",
            )

            plt.title("Channel Phase Correction")
            plt.xlabel("Phase [rad]")
            plt.ylabel("Frequency [Hz]")
            plt.legend(loc="lower left")

            plt.show()

        return phase_linear_trend

    def sync_correct(self):
        phase_trend = self._find_drift(plot=True)
        self.H_est *= np.exp(-1.0j * phase_trend)

    def plot_channel(self):
        _, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5))

        freqs = np.linspace(0, FS, self.H_est.shape[0])

        ax_left.plot(freqs, 10 * np.log10(np.abs(self.H_est)))
        ax_left.set_xlabel("Frequency [Hz]")
        ax_left.set_ylabel("Magnitude [dB]")

        ax_right.scatter(freqs, self._unwrap_phases(), marker=".")
        ax_right.set_xlabel("Frequency [Hz]")
        ax_right.set_ylabel("Phase [rad]")

        print(np.angle(self.H_est))
        plt.show()

    def plot_decoded_symbols(self, i=-1):
        # Plots decoded symbols coloured by frequency bin
        X = self.Xhats[i]
        plt.scatter(
            X.real, X.imag, c=np.arange(len(X)), cmap="gist_rainbow_r", marker="."
        )

        plt.title("Decoded constellation symbols")
        plt.axhline(0, color="black", linestyle=":")
        plt.axvline(0, color="black", linestyle=":")
        plt.xlabel("Re")
        plt.ylabel("Im")
        cbar = plt.colorbar()
        cbar.set_label("Subcarrier number")

        plt.show()

    def mse_decode(self, i=-1):
        Xhat = self.Xhats[i]
        X = self.source_chunks[i][FREQ_MIN:FREQ_MAX]
        breakpoint()


n = 25
source = np.random.choice(VALUES, N_BINS * n)

np.seterr(all="ignore")  # Supresses runtime warnings

transmission = Transmission(source)
transmission.record_signal(afplay=True)
transmission.save_signals()
# transmission.load_signals()

# Initial synchronisation
transmission.synchronise(plot=True)
transmission.estimate_H()
transmission.estimate_Xhats()
transmission.plot_channel()
transmission.plot_decoded_symbols()

# Correct synchronisation for drift
# print("2nd pass:")
transmission.sync_correct()
# transmission.estimate_H()
# transmission.estimate_Xhats()
transmission.plot_channel()
# transmission.plot_decoded_symbols()
transmission.mse_decode()


"""
- Adjust OFDM symbol indicies from samples extra/missing (DONE)
- Change estimation to average magnitudes and angles separately (DONE)
- Add a function to check the error rate between the transmitted and received constellations
    - Use decoder?
"""
