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

    def synchronise(self, offset=5, plot=False):
        # End of chirp is half a second after chirp.
        # Offset gives the number of samples to shift back by to ensure you're sampling early.
        peaks = self._find_chirp_peaks()
        frame_start_index = peaks.min() + self.fs // 2 - offset
        frame_end_index = peaks.max() - self.fs // 2 - offset

        num_symbols = np.round(
            (frame_end_index - frame_start_index) / (self.L + self.N)
        )

        sampling_error = (
            frame_end_index - frame_start_index - num_symbols * (self.L + self.N)
        )

        print(
            "Peaks: {}\nFrame start: {}\nFrame end: {}\nNum symbols: {}\nSampling error: {}".format(
                peaks, frame_start_index, frame_end_index, num_symbols, sampling_error
            )
        )

        self.Rs = self._identify_Rs(frame_start_index, num_symbols, sampling_error)

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
            np.flip(self.chirp.signal),
            mode="same",
        )
        peaks = sig.find_peaks(
            conv, height=0.5 * np.abs(conv).max(), distance=transmission.fs - 1
        )[0]
        return peaks

    def _identify_Rs(self, frame_start_index, num_symbols, sampling_error):
        Rs = []
        error_per_symbols = sampling_error / num_symbols

        for i in range(n):
            start = int(
                frame_start_index
                + self.L
                + np.rint((self.L + self.N + error_per_symbols) * i)
            )
            end = start + self.N
            r = self.received_signal[start:end]
            R = np.fft.fft(r)
            Rs.append(R)
        return Rs

    def estimate_H(self):
        # Returns a channel estimate from the known and received OFDM symbols
        R = np.vstack(self.Rs[:4])
        X = np.vstack(self.Xs[:4])
        
        # self.H_est = np.mean(R / X, axis=0)
        
        magnitudes = np.mean(np.abs(R / X), axis=0)
        angles = np.mean(np.angle(R / X), axis=0)
        self.H_est = magnitudes * np.exp(1j * angles)

    def estimate_Xhats(self):
        self.Xhats = []

        for R in self.Rs:
            X = R / self.H_est

            # Only add useful part of carrier data X
            # Bins determined by standard (86 to 854 inclusive)
            self.Xhats.append(X[86:854])

    def _unwrap_phases(self):
        # Correct for wrapping of phases to [pi, -pi]
        phases = np.angle(self.H_est)
        for i in range(phases.shape[0] - 1):
            diff = phases[i + 1] - phases[i]
            if diff >= np.pi:
                phases[i + 1 :] -= 2 * np.pi
            elif diff <= np.pi:
                phases[i + 1 :] += 2 * np.pi

        return phases

    def _find_drift(self, plot=False):
        # Correct for wrapping of phases to [pi, -pi]
        phases = self._unwrap_phases()

        # Correct for linear trend by adjusting so that corrected_phases[-1] = -corrected_phases[1] for conjugacy
        # phase_linear_trend = np.linspace(0, phases[-1], phases.shape[0])
        grad = (phases[1]+phases[-1])/self.N
        phase_linear_trend = grad * np.arange(0, self.N)
        corrected_phases = phases - phase_linear_trend

        assert np.allclose(corrected_phases[1:self.N//2], -np.flip(corrected_phases[1+self.N//2:])) == True     # Assert conjugacy
        
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
                freqs,
                phases,
                color="green",
                marker=".",
                label="Unwrapped phases"
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
            plt.ylabel("Phase [rad]")
            plt.xlabel("Frequency [Hz]")
            plt.legend(loc="lower left")

            plt.show()

        return corrected_phases

    def sync_correct(self):
        correct_phases = self._find_drift()
        self.H_est = np.abs(self.H_est) * np.exp(1.j*correct_phases)

    def plot_channel(self):
        _, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5))

        freqs = np.linspace(0, self.fs, self.H_est.shape[0])

        ax_left.plot(freqs, 10 * np.log10(np.abs(self.H_est)))
        ax_left.set_xlabel("Frequency [Hz]")
        ax_left.set_ylabel("Magnitude [dB]")

        print(np.angle(self.H_est)[1:])
        ax_right.scatter(freqs, self._unwrap_phases(), marker=".")
        ax_right.set_xlabel("Frequency [Hz]")
        ax_right.set_ylabel("Phase [rad]")

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


fs = 48000
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

# Known OFDM symbols from the standard
known_symbol = np.load("known_ofdm_symbol.npy")[1 : 4096 // 2]
n = 25
source = np.tile(known_symbol, n)

np.seterr(all="ignore")  # Supresses runtime warnings

transmission = Transmission(source, constellation_map, fs=fs)
# transmission.record_signal()
# transmission.save_signals()
transmission.load_signals()

# Initial synchronisation
transmission.synchronise()
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
transmission.plot_decoded_symbols()


"""
- Adjust OFDM symbol indicies from samples extra/missing (DONE)
- Change estimation to average magnitudes and angles separately
- Add a function to check the error rate between the transmitted and received constellations
    - Use decoder?
"""
