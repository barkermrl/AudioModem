from pathlib import Path
from bitarray import bitarray
import struct
import numpy as np
import ldpc_jossy.py.ldpc as ldpc

CONSTELLATION_MAP = {
    "00": complex(1 / np.sqrt(2), 1 / np.sqrt(2)),
    "01": complex(-1 / np.sqrt(2), 1 / np.sqrt(2)),
    "11": complex(-1 / np.sqrt(2), -1 / np.sqrt(2)),
    "10": complex(1 / np.sqrt(2), -1 / np.sqrt(2)),
}


class createFile:
    def __init__(self, tx_file: str):
        self.filename = tx_file

        self.header = self._create_header()
        self.data = self._get_data()

        self.padded = self._pad_for_encoder()

    def _create_header(self):
        """
        Header contains:
            File length L       unsigned integer        4 bytes little-endian
            File name           ascii characters        variable no of bytes
            Null terminator                             1 byte
        then L bytes of data
        """

        # Length of data
        filelength_bytes = Path(self.filename).stat().st_size
        encoded_filelength = struct.pack(
            "<L", filelength_bytes
        )  # "<L" is 4 byte little endian

        # Filename in ascii
        encoded_filename = self.filename.encode(encoding="ascii", errors="ignore")

        # Null terminator
        null = "\0".encode(encoding="ascii", errors="ignore")

        header = encoded_filelength + encoded_filename + null

        return header

    def _get_data(self):
        """
        Turn file data into L bytes
        """
        file = open(self.filename, "rb")
        data = file.read()
        file.close()

        return data

    def _pad_for_encoder(self):
        """
        Pad transmitted data for input to the encoder
        Must be [ ENCODER CONDITIONS ]
        """
        file = self.header + self.data

        padded_file = file

        return padded_file


def save_bits():
    f = createFile("group5.ppm")
    pf = f.padded

    a = bitarray(endian="little")
    a.frombytes(pf)
    bits = a.tolist()

    c = ldpc.code(standard="802.16", z=64, rate="1/2")

    num_extra_bits = (c.K - len(bits) % c.K) % c.K
    bits.extend([0] * num_extra_bits)

    source = [bits[i : i + c.K] for i in range(0, len(bits), c.K)]

    codewords = []

    for row in source:
        codewords.extend(list(c.encode(np.array(row))))

    assert len(codewords) / len(bits) == 2
    codewords = np.array(codewords)
    np.save("frenzy.npy", codewords)


def save_constellation_values():
    codewords = np.load("frenzy.npy")
    num_values = len(codewords) // 2
    constellation_values = np.zeros(num_values, dtype=complex)
    for i in range(num_values):
        key = str(codewords[2 * i]) + str(codewords[2 * i + 1])
        constellation_values[i] = CONSTELLATION_MAP[key]

    constellation_values = np.array(constellation_values)
    np.save("frenzy_constellation_values.npy", codewords)


save_bits()
save_constellation_values()
