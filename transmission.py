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
                    np.random.choice(VALUES, FREQ_MIN - 1),
                    source_chunk,
                    np.random.choice(VALUES, NUM_CARRIERS - FREQ_MAX + 1),
                )
            )
            assert len(data_carriers) == NUM_CARRIERS
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
        # np.testing.assert_equal(self.signal, signal)
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

    def synchronise(self, offset=0, plot=False):
        # End of chirp is half a second after chirp.
        # Offset gives the number of samples to shift back by to ensure you're sampling early.
        # Peaks should be length 4 (1 chirp at the start and end of signal)
        # In addition, each frame starts and ends with a chirp.

        peaks = np.sort(self._find_chirp_peaks())
        assert len(peaks) == 4, f"Found {len(peaks)} peaks, expected 4"

        # Get starting known OFDM blocks from chirp synchronisation
        first_known_ofdm_start_index = peaks[1] - offset
        first_known_ofdm_end_index = (
            first_known_ofdm_start_index + len(PREAMBLE) - len(CHIRP)
        )

        # Get ending known OFDM blocks from chirp synchronisation
        last_known_ofdm_end_index = peaks[-2] - len(CHIRP) - offset
        last_known_ofdm_start_index = (
            last_known_ofdm_end_index - len(ENDAMBLE) + len(CHIRP)
        )

        # Find the number of symbols from the samples between the known OFDM start and end indices
        num_symbols = int(
            np.round(
                (last_known_ofdm_start_index - first_known_ofdm_end_index) / (L + N)
            )
        )

        # Estimate where the ending known OFDM blocks are from the starting blocks and number of symbols
        last_known_ofdm_start_index_est = first_known_ofdm_end_index + num_symbols * (
            L + N
        )
        last_known_ofdm_end_index_est = (
            last_known_ofdm_start_index_est + len(ENDAMBLE) - len(CHIRP)
        )

        # Find the sampling error from the difference in the location of the last known OFDM block
        # A negative error means we've received too few samples
        error = last_known_ofdm_start_index - last_known_ofdm_start_index_est
        error_per_sample = error / (peaks[-2] - peaks[1])
        error_per_sample *= 1.1
        print(error_per_sample)

        self.drift_per_sample = error_per_sample

        print(
            f"""
            Peaks: {peaks}
            Num symbols: {num_symbols}
            First OFDM block: {first_known_ofdm_start_index} to {first_known_ofdm_end_index}
            Last OFDM block (length): {last_known_ofdm_start_index_est} to {last_known_ofdm_end_index_est}
            Last OFDM block (chirp): {last_known_ofdm_start_index} to {last_known_ofdm_end_index}            
            Sampling error (-ve means samples missed): {error}
        """
        )

        if plot:
            plt.plot(self.received_signal)

            plt.axvline(
                first_known_ofdm_start_index,
                color="r",
                linestyle=":",
                label="First OFDM block",
            )
            plt.axvline(first_known_ofdm_end_index, color="r", linestyle=":")
            plt.axvline(
                last_known_ofdm_start_index,
                color="g",
                linestyle=":",
                label="Last OFDM block: chirp",
            )
            plt.axvline(last_known_ofdm_end_index, color="g", linestyle=":")
            plt.axvline(
                last_known_ofdm_start_index_est,
                color="y",
                linestyle=":",
                label="Last OFDM block: length",
            )
            plt.axvline(last_known_ofdm_end_index_est, color="y", linestyle=":")

            plt.title("Received signal")
            plt.xlabel("Sample index")
            plt.ylabel("Amplitude")
            plt.legend(loc="upper right")

            plt.show()

        # Pull out the known OFDM blocks from the received signal
        self.known_symbols_start = self.received_signal[
            first_known_ofdm_start_index + L : first_known_ofdm_end_index
        ]
        self.known_symbols_end = self.received_signal[
            last_known_ofdm_start_index + L : last_known_ofdm_end_index
        ]
        self.known_symbols_end_est = self.received_signal[
            last_known_ofdm_start_index_est + L : last_known_ofdm_end_index_est
        ]

        # Pull out received OFDM block
        self.Rs = self._identify_Rs(
            first_known_ofdm_end_index, num_symbols
        )
    
    def _find_chirp_peaks(self):
        # window = lambda x: x * np.hanning(len(x))
        conv = sig.convolve(
            self.received_signal,
            np.flip(CHIRP),
        )
        peaks = sig.find_peaks(conv, height=0.5 * np.abs(conv).max(), distance=FS // 2)[
            0
        ]
        return peaks

    def _identify_Rs(self, first_known_ofdm_end_index, num_symbols):
        Rs = []
        # 0 drift at the end of the chirp
        drift_at_known_ofdm_end = int( np.round( self.drift_per_sample * (L + len(self.known_symbols_start)) ) )
        drift_per_symbol = self.drift_per_sample * (L + N)

        for i in range(num_symbols):
            start = (
                first_known_ofdm_end_index
                + L
                + drift_at_known_ofdm_end
                + int(np.round((L + N + drift_per_symbol) * i))
            )
            end = start + N
            r = self.received_signal[start:end]
            R = np.fft.fft(r)
            Rs.append(R)
        return Rs

    def estimate_H(self):
        # Returns a channel estimate from the known and received OFDM symbols
        H_est_start = self._complex_average(self.known_symbols_start)
        H_est_end = self._complex_average(self.known_symbols_end)
        self.H_est = H_est_start

    def _complex_average(self, known_symbols, plot_each=False):
        r = known_symbols.reshape((-1, N))
        R = np.fft.fft(r)
        # magnitudes = np.mean(np.abs(R / KNOWN_SYMBOL_BIG_X), axis=0)
        # angles = np.mean(np.angle(R / KNOWN_SYMBOL_BIG_X), axis=0)
        # return magnitudes * np.exp(1j * angles)

        temp = np.concatenate([
            [0],
            np.arange(1, N // 2) / N,
            [0],
            np.arange(1-N//2, 0) / N
        ])

        # temp = np.arange(0, N)/N
        # temp[N//2] = 0

        for i, R_block in enumerate(R):
            drift_at_block_start = self.drift_per_sample * (L + N * i)
            drift_correction = np.exp(2j * np.pi * drift_at_block_start * temp)
            R_block *= drift_correction

        H_est = np.mean(R / KNOWN_SYMBOL_BIG_X, axis=0)

        if plot_each:
            freqs = np.linspace(0, FS, N)
            fig, axs = plt.subplots(2, 5)
            for i in range(axs.shape[1] - 1):
                axs[0, i].set_title(f"H_est from known OFDM {i}")

                axs[0, i].plot(freqs, 10 * np.log10(np.abs(H_est_matrix[i, :])))
                axs[0, i].set_xlabel("Freq. [Hz]")
                axs[0, i].set_ylabel("Mag. [dB]")

                axs[1, i].scatter(freqs, np.angle(H_est_matrix[i, :]), marker=".")
                axs[1, i].set_xlabel("Freq. [Hz]")
                axs[1, i].set_ylabel("Phase [rad]")

            axs[0, 4].set_title("H_est from average")

            axs[0, 4].plot(freqs, 10 * np.log10(np.abs(H_est)))
            axs[0, 4].set_xlabel("Freq. [Hz]")
            axs[0, 4].set_ylabel("Phase [rad]")

            axs[1, 4].scatter(freqs, np.angle(H_est_matrix[i, :]), marker=".")
            axs[1, 4].set_xlabel("Freq. [Hz]")
            axs[1, 4].set_ylabel("Phase [rad]")
            plt.show()

        return H_est

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

    def _check_decoding(self, i):
        get_sign_tuple = lambda x: (np.sign(x.real), np.sign(x.imag))
        num_correct = 0

        Xhat = self.Xhats[i]
        X = self.source_chunks[i]
        assert len(Xhat) == len(X)

        for i in range(len(Xhat)):
            if get_sign_tuple(Xhat[i]) == get_sign_tuple(X[i]):
                num_correct += 1

        proportion_correct = num_correct / len(Xhat)

        # breakpoint()
        return proportion_correct

    def sync_correct(self, plot=False):
        phase_trend = self._find_drift(plot)
        self.H_est *= np.exp(-1.0j * phase_trend)

    def plot_channel(self):
        _, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5))

        freqs = np.linspace(0, FS, self.H_est.shape[0])

        ax_left.plot(freqs, 10 * np.log10(np.abs(self.H_est)))
        ax_left.set_xlabel("Frequency [Hz]")
        ax_left.set_ylabel("Magnitude [dB]")

        ax_right.scatter(freqs, np.angle(self.H_est), marker=".")
        ax_right.set_xlabel("Frequency [Hz]")
        ax_right.set_ylabel("Phase [rad]")

        plt.title("Estimated Channel")
        plt.show()

    def plot_decoded_symbols(self, i=-1):
        # Find the proportion of decoded values in the correct quadrant
        proportion_correct = self._check_decoding(i=i)

        # Plots decoded symbols coloured by frequency bin
        X = self.Xhats[i]
        plt.scatter(
            X.real, X.imag, c=np.arange(len(X)), cmap="gist_rainbow_r", marker="."
        )

        plt.title(
            f"Decoded constellation values for symbol {i}\nProportion correct = {proportion_correct}"
        )
        plt.axhline(0, color="black", linestyle=":")
        plt.axvline(0, color="black", linestyle=":")
        plt.xlabel("Re")
        plt.ylabel("Im")
        cbar = plt.colorbar()
        cbar.set_label("Subcarrier number")

        plt.show()

    def absolute_violation(self):
        # Change the drift per sample by some factor
        # Optimise drift factor by looking at the decoded correctly probability over the known symbols



n = 25
np.random.seed(0)
source = np.random.choice(VALUES, N_BINS * n)

np.seterr(all="ignore")  # Supresses runtime warnings

transmission = Transmission(source)
# transmission.record_signal()
# transmission.save_signals()
transmission.load_signals()

# Initial synchronisation
transmission.synchronise(plot=True)
transmission.estimate_H()
transmission.estimate_Xhats()
transmission.plot_channel()
transmission.plot_decoded_symbols()
# transmission.mse_decode()

print("Block no. -- Frac. decoded correctly")
for i in range(n):
    print(i, "          ", transmission._check_decoding(i=i))

# Correct synchronisation for drift
# print("2nd pass:")
# transmission.sync_correct()
# transmission.estimate_H()
# transmission.estimate_Xhats()
# transmission.plot_channel()
# transmission.plot_decoded_symbols()


"""
- Adjust OFDM symbol indicies from samples extra/missing (DONE)
- Change estimation to average magnitudes and angles separately (DONE)
- Add a function to check the error rate between the transmitted and received constellations
    - Use decoder?
"""
