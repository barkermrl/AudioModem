from dataclasses import replace
from math import remainder
from pickle import FRAME
import numpy as np
import LDPC_utils as lp
import scipy.signal as sig
from bitarray import bitarray
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import sounddevice as sd

# Standard
FS = 48000
L = 512
N = 4096
FRAME_LENGTH = 128
NUM_FRAME_VALUES = FRAME_LENGTH * (L + N)
FREQ_MIN = 86
FREQ_MAX = 854

# Get gap
gap = np.zeros(FS)

# Get source bits
lorem = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Praesent ut risus ligula. Pellentesque quis ultrices arcu. Maecenas vulputate, ante vel mattis faucibus, lacus ex congue erat, at elementum nunc urna vitae eros. Donec id ultricies dui, eu varius eros. In luctus, mi sit amet sodales tincidunt, sem ligula bibendum elit, ut porta purus dui non lacus. Vivamus id eros maximus, dictum eros nec, lacinia justo. Aliquam lobortis vulputate libero vitae ornare. Proin luctus sed odio vel euismod. Duis scelerisque fringilla eleifend. Praesent orci nulla, tristique id magna in, varius pellentesque metus. Cras venenatis vitae ante eu ullamcorper. Mauris ac nisi convallis, malesuada mi quis, pellentesque sem. Maecenas ut ipsum at ligula congue efficitur eu at neque. Duis et egestas metus. Vivamus placerat purus tortor, a facilisis nibh malesuada id. Etiam sit amet lacinia mauris. Donec id placerat est. Mauris efficitur sed est sed ultricies. Vestibulum blandit condimentum arcu, in placerat risus. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Suspendisse consequat turpis magna, at fermentum velit pellentesque eget. Proin facilisis magna sed orci ultrices, vitae scelerisque libero pellentesque. Ut ex tortor, vestibulum sed dictum sed, volutpat et nisi. Cras in tincidunt tellus, ac pellentesque elit. Praesent ut turpis dignissim, interdum neque sed, dapibus leo. Nullam eu nisl est. Vestibulum commodo congue erat. Vestibulum sollicitudin purus sit amet nunc mattis, sit amet tempus mi faucibus. Phasellus imperdiet urna non sapien scelerisque, nec pretium ex aliquet. Nullam quis diam purus. Suspendisse velit orci, aliquet non tellus ut, porttitor volutpat velit. Nunc at ex molestie, vestibulum nisi id, pharetra lacus. Morbi eu volutpat sem. Etiam ullamcorper nunc a quam venenatis, nec varius magna posuere. Duis quis facilisis lorem. Etiam sed volutpat ligula. Nam ex ex, rhoncus id scelerisque vel, ultrices ac odio. Aenean sodales vulputate pretium. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Donec cursus dignissim ante, et condimentum diam interdum vitae. Sed porta mi sed risus ultricies, sit amet consequat augue mollis. Sed lacinia velit finibus, tincidunt diam eu, aliquet felis. Phasellus suscipit elementum rutrum. Nam pellentesque tincidunt ornare. Nunc feugiat placerat est et rutrum. Integer libero sapien, mollis ac leo vel, iaculis mattis augue. Donec non posuere arcu. Nam condimentum sagittis est et aliquam. Nunc tincidunt enim lectus, non euismod tortor pulvinar eu. Aliquam lobortis lectus eu tristique condimentum. Fusce quis tellus non turpis sollicitudin ornare vel eu purus. Nullam id diam ut libero tincidunt blandit. Curabitur et velit metus. Duis vel tristique massa, ac congue nisl. Cras sodales pharetra massa, at placerat enim sagittis nec. Sed aliquam nec mauris et ultrices. Nulla est est, congue non orci quis, pellentesque fringilla sapien. Nunc congue porta erat. Sed consectetur malesuada risus quis porttitor. Aenean tempor mauris fermentum consectetur pellentesque. Donec maximus laoreet quam. Morbi ac leo ante. Pellentesque commodo pulvinar ultrices. Donec suscipit vitae nulla ac dignissim. Morbi justo purus, fringilla sit amet sodales vitae, scelerisque id diam. Nulla in suscipit urna. Phasellus et mi sit amet eros aliquam facilisis id a nibh. In vestibulum dictum erat a consequat."
lorem_bytes = bytes(lorem, encoding="utf-8")
a = bitarray(endian = 'little')
a.frombytes(lorem_bytes)
source_bits = []
for i in a:
    source_bits.append(i)
source_bits = np.array(source_bits)

# Encode source bits
c = lp.get_code()
x = lp.encode(source_bits, c)

# Get known OFDM
known_ofdm_block = np.load("known_ofdm_symbol.npy")
known_ofdm_symbol = np.fft.ifft(known_ofdm_block)
known_ofdm_symbol *= 1/np.max(known_ofdm_symbol)
guard = known_ofdm_block[-L:]

# Get Chirp
t = np.linspace(0, 1, FS)
chirp = sig.chirp(t, 1e3, 1, 10e3, method="linear")

# Build prefix
prefix = np.concatenate([
    chirp, guard, known_ofdm_symbol,known_ofdm_symbol,known_ofdm_symbol,known_ofdm_symbol
])

# Build endfix
endfix = np.concatenate([
    guard, known_ofdm_symbol,known_ofdm_symbol,known_ofdm_symbol,known_ofdm_symbol, chirp
])

# Convert x into constelation values:
CONSTELLATION_MAP = {
    "00": complex(1 / np.sqrt(2), 1 / np.sqrt(2)),
    "01": complex(-1 / np.sqrt(2), 1 / np.sqrt(2)),
    "11": complex(-1 / np.sqrt(2), -1 / np.sqrt(2)),
    "10": complex(1 / np.sqrt(2), -1 / np.sqrt(2)),
}

const = []
for i in range(len(x)//2):
    const.append(CONSTELLATION_MAP[str(int(x[2*i])) + str(int(x[2*i + 1]))])

# Pad constelation values with random const.symbs:
symbs_per_block = FREQ_MAX - FREQ_MIN
remainder = len(const) % symbs_per_block
num_symbs = symbs_per_block - remainder
if num_symbs != 0:
    padding = np.random.choice(list(CONSTELLATION_MAP.values()), num_symbs, replace = True).tolist()
const += padding

# Split constelations into blocks
num_blocks = len(const)//symbs_per_block
blocks = []
for i in range(num_blocks):
    block_start_padding = np.random.choice(list(CONSTELLATION_MAP.values()), FREQ_MIN-1, replace = True)
    end_padding_len = N - (2*symbs_per_block + 2 + 2*len(block_start_padding))
    block_end_padding = np.random.choice(list(CONSTELLATION_MAP.values()), end_padding_len, replace = True)
    block = np.concatenate([
        [0],block_start_padding.tolist(),const[i*(symbs_per_block): (i+1)*(symbs_per_block)], [0], np.conjugate(const[i*(symbs_per_block): (i+1)*(symbs_per_block)]).tolist(), np.conjugate(block_start_padding).tolist()
    , block_end_padding])
    symbol = np.fft.fft(block)
    symbol = np.concatenate([
        block[-L:], block
    ])
    symbol *= 1/np.max(symbol)
    blocks.append(symbol)

print(f'Number of blocks {len(blocks)}')

# Build frames:
frames = []
num_frame = int(np.ceil(len(blocks)/FRAME_LENGTH))
for i in range(num_frame):
    frame_blocks = blocks[i*FRAME_LENGTH: (i+1)*FRAME_LENGTH]
    ls = []
    for symb in frame_blocks:
        ls += symb.tolist()
    frames.append(ls)

# Build signal from frames:
signal = []
signal += gap.tolist()
signal += chirp.tolist()

for frame in frames:
    #print(frame)
    signal += prefix.tolist()
    signal += frame
    signal += endfix.tolist()

signal += chirp.tolist()
plt.plot(signal)
plt.show()

#wav.write('Lorem_test.wav', FS, np.array(signal).real)
print('Saved')

def record_file(length, name):
    print("Recording...")
    r = sd.rec(
        int(length * FS),
        samplerate=FS,
        channels=1,
        blocking=True,
    ).flatten()
    print("Recording finished")
    np.save(f'{name}.npy', r)
    print(f"Saved as {name}.npy")

record_file(15, 'lorem_rec')
