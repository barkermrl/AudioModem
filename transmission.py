# Import libraries
import matplotlib.pyplot as plt
import numpy as np
from pytest import FixtureRequest
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


class Transmission:
    def __init__(self, source):
        self.source = source
        self.source_chunks = source.reshape((-1, N_BINS))

        # Create signal
        symbols = self._create_symbols()
        frames = self._create_frames(symbols)

        self.signal = np.concatenate((START, frames, END))

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

    def _create_frames(self, symbols):
        num_whole_frames = len(symbols) // NUM_FRAME_VALUES

        frames = []
        for i in range(num_whole_frames):
            frame = symbols[i * NUM_FRAME_VALUES : (i + 1) * NUM_FRAME_VALUES]
            frames.append(np.concatenate([PREAMBLE, frame, ENDAMBLE]))

        final_frame = symbols[num_whole_frames * NUM_FRAME_VALUES :]
        frames.append(np.concatenate([PREAMBLE, final_frame, ENDAMBLE]))

        return np.concatenate(frames)

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

    def play_signal(self, afplay=False):
        if afplay:
            # Requires afplay to be installed as a terminal command!
            wav.write("tmp.wav", FS, self.signal)
            subprocess.Popen(["afplay", "tmp.wav"])
        else:
            wav.write("frenzy_signal.wav", FS, self.signal)
            sd.play("frenzy_signal.wav", FS, blocking=True)

    def record_signal(self):
        print("Recording...")
        self.received_signal = sd.rec(
            int(15 * FS),
            samplerate=FS,
            channels=1,
            blocking=True,
        ).flatten()
        print("Recording finished")

    def get_frames(self):
        peaks = self._find_chirp_peaks() - len(CHIRP)
        plt.plot(self.received_signal)
        plt.vlines(peaks, 0, np.max(self.received_signal), color = 'r')
        plt.show()
        print(f"Number of peaks found = {len(peaks)}")
        
        num_frames = (len(peaks) - 2) / 2
        print(f"Number of frames = {num_frames}")
        frames = []
        # Extract each frame
        for i in range(int(num_frames)):
            frame = self.received_signal[peaks[2*i] : peaks[2*i + 3] + len(CHIRP)]
            frames.append(frame)
        print(f'Number of frames extracted = {len(frames)}')
        # frames is a list of np.ndarrays
        return frames

    def synchronise(self, offset=0, print_results=False, plot=False):
        # End of chirp is half a second after chirp.
        # Offset gives the number of samples to shift back by to ensure you're sampling early.
        # Peaks should be length 4 (1 chirp at the start and end of signal)
        # In addition, each frame starts and ends with a chirp.

        peaks = np.sort(self._find_chirp_peaks())
        assert len(peaks) == 4, f"Found {len(peaks)} peaks, expected 4"

        # Find the number of symbols from the samples between the known OFDM start and end indices
        num_symbols = int(
            np.round(
                (peaks[2] - peaks[1] - len(ENDAMBLE) - len(PREAMBLE) + len(CHIRP))
                / (L + N)
            )
        )
        self.num_symbols = num_symbols
        print(f"Number of symbols in frame {num_symbols}")

        if print_results:
            print(
                f"""
                Peaks: {peaks}
                Num symbols: {num_symbols}
                First OFDM block: {first_known_ofdm_start_index} to {first_known_ofdm_end_index}
                Last OFDM block (length): {last_known_ofdm_start_index_est} to {last_known_ofdm_end_index_est}
                Last OFDM block (chirp): {last_known_ofdm_start_index} to {last_known_ofdm_end_index}            
                Sampling error (-ve means samples missed): {error}
                Drift per sample: {error_per_sample}
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

        # Resample
        expected_length_signal = (
            len(PREAMBLE) - len(CHIRP) + num_symbols * (L + N) + len(ENDAMBLE)
        )
        self.resampled_signal = sig.resample(
            self.received_signal[peaks[1] : peaks[2]], expected_length_signal
        )

        # Pull out received OFDM block
        self.known_ofdm_symbols_start = self.resampled_signal[L : L + 4 * N]
        self.known_ofdm_symbols_end = self.resampled_signal[-len(ENDAMBLE) + L:-len(CHIRP)]
        print(f"Number of known symbols {len(self.known_ofdm_symbols_end)/N}")
        self.data_ofdm_symbols = self.resampled_signal[L + 4 * N : -len(ENDAMBLE)] # including guards
        self.Rs = self._identify_Rs(num_symbols)

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

    def _identify_Rs(self, num_symbols):
        Rs = []
        for i in range(num_symbols):
            start = L + i * (L + N)
            end = start + N
            r = self.data_ofdm_symbols[start:end]
            R = np.fft.fft(r)
            Rs.append(R)
        return Rs

    def estimate_H(self):
        # Returns a channel estimate from the known and received OFDM symbols
        first_known_symbol = self.known_ofdm_symbols_start.reshape((-1, N))
        last_known_symbol = self.known_ofdm_symbols_end.reshape((-1,N))
        dft_list = []
        for first, last in zip(first_known_symbol, last_known_symbol):
            first_dft = np.fft.fft(first, N)
            last_dft = np.fft.fft(last,N)
            dft_list.append(first_dft)
            dft_list.append(last_dft)
        
        self.H_est = np.mean(dft_list, axis = 0)/KNOWN_SYMBOL_BIG_X
        self.vars = np.var(dft_list, axis=0)
        self.vars = np.tile(self.vars[FREQ_MIN:FREQ_MAX].real, self.num_symbols)

    def Xhats_estimate(self):
        self.Xhats = []
        ofdm_blocks = np.split(self.data_ofdm_symbols, self.num_symbols)
        for block in ofdm_blocks:
            block = block[L:]
            block_dft = np.fft.fft(block)
            equalized_dft = block_dft/self.H_est
            ofdm_symbs = equalized_dft[FREQ_MIN:FREQ_MAX].tolist()
            self.Xhats += [ofdm_symbs]

    def split_list(self, data, length):
        index = 0
        data = data.tolist()
        ls = []
        while index < len(data):
            ls.append(data[index : index + length])
            index += length
        return ls

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
        # Optimise drift factor by looking at the decoded correctly proportion over the known symbols

        def optimise_over_known_OFDM(factor):
            # Returns a channel estimate from the known and received OFDM symbols
            # Also adjusts for drift factor
            r_start = self.known_symbols_start.reshape((-1, N))
            R_start = np.fft.fft(r)

            temp = np.concatenate(
                [[0], np.arange(1, N // 2) / N, [0], np.arange(1 - N // 2, 0) / N]
            )

            for i, R_block in enumerate(R_start):
                drift_at_block_start = self.drift_per_sample * factor * (L + N * i)
                drift_correction = np.exp(2j * np.pi * drift_at_block_start * temp)
                R_block *= drift_correction

            for i, R_block in enumerate(R_end):
                drift_at_block_start = self.drift_per_sample * factor * (L + N * i)
                drift_correction = np.exp(2j * np.pi * drift_at_block_start * temp)
                R_block *= drift_correction

            H_est = np.mean(R_start / KNOWN_SYMBOL_BIG_X, axis=0)

            Xhat = np.mean(R_start, axis=0) / H_est

            X_diffs = Xhat - KNOWN_SYMBOL_BIG_X
            mmse = np.mean(X_diffs * np.conjugate(X_diffs))

            return mmse

        res = minimize(optimise_over_known_OFDM, 1.1, method="Nelder-Mead")
        factor_opt = res.x
        mmse_opt = res.fun
        print(factor_opt, mmse_opt)

        self.drift_per_sample *= factor_opt
