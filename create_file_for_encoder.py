from pathlib import Path
import struct

class createFile:
    def __init__(self, tx_file : str):
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
        filelength_bytes = Path( self.filename ).stat().st_size
        encoded_filelength = struct.pack( "<L", filelength_bytes )  # "<L" is 4 byte little endian

        # Filename in ascii
        encoded_filename = self.filename.encode( encoding="ascii", errors="ignore" )

        # Null terminator
        null = "\0".encode( encoding="ascii", errors="ignore" )

        header = encoded_filelength + encoded_filename + null

        return header

    def _get_data(self):
        """
        Turn file data into L bytes
        """
        file = open( self.filename, "rb" )
        data = file.read()
        file.close()

        data = data

        return data

    def _pad_for_encoder(self):
        """
        Pad transmitted data for input to the encoder
        Must be [ ENCODER CONDITIONS ]
        """
        file = self.header + self.data

        padded_file = file

        return padded_file

f = createFile("frenzy.ppm")
pf = f.padded

print(pf)