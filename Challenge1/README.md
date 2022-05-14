Welcome to the Weekend 1 Challenge. This takes the form of a quiz that you are meant to take by group (one student answers for their group.) Unlimited attempts are allowed and you are expected to try to get all the questions right (though it may not be easy...)

9 files have been modulated using OFDM (full details below) and passed through a channel with a known impulise response (the same for all files). It is your task to recover the files, and guess a 5-letter word that somehow relates to the file. One of the files has an extra second 5-letter word for an extra bonus. 

First, dowlnoad the channel impulse response here. 

* each file was modulated as binary data and preceded by a header that contains a ‘\0’ terminated string with the file name, and another ‘\0’ terminated string with the file size (number of bytes.) After the second ‘\0’ it’s raw file data for the number of bytes specified in the file size string.

* Bytes were mapped to QPSK constellations 2 bits at a time (starting from the most significant 2 bytes and ending with the least significant 2 bytes) using Gray coding (anti-clockwise starting from pi/4 bit pairs 00, 01, 11, 10).
Note: I am not 100% confident about the ordering of the Gray coding. You may want to check 00 10 11 01 just in case…
Note 2: if any of you write “Grey” like some lecturers in Cambridge who feel that they prefer to use British spelling, please aware that Frank Gray from Bell Labs who came up with this mapping spelled his name “Gray” irrespective of which continent he happened to be on… Please don’t join the those misinformed lecturers who want to impose British spelling on his family name

* An IDFT size of 1024 was applied with a cyclic prefix of length 32. Information is only mapped to frequency bins 1 to 511. Frequency bins 513 to 1023 contain the reverse ordered conjugate complex values of frequency bins 1 to 511. Frequency bins 0 and 512 contain 0 (no information, value 0.) This all ensures that the output of the OFDM modulator is a real (baseband) vector.

* 2 bit groups were added at the end of the data to complete an integer factor of 511. Those extra 2 bit groups are all 00 (note that this is not a good idea for your final modem!)

* A cyclic prefix of 32 was appended before each DFT block and the resulting blocks of 1056 values were concatenated into one long signal.

* In OFDM terminology, a "block of data whose iDFT/DFT  you take then add a cyclic prefix" is confusingly called an "OFDM symbol". We will call the constellation symbols that make up the block "constellation values" to avoid confusion with "OFDM symbols". Finally, avoid speaking of "blocks" at this stage because later in the standardisation you may think of "blocks" and "frames" comprising several OFDM symbols with an agreed prefix for synchronisation and channel estimation.

* In this exercise, there is no need for synchronisation or channel estimation: the channel is known and the first OFDM symbols starts with the first received sample in the data I am supplying.

* This signal was fed through an ISI channel consisting of an FIR with the impulse response you have been given. The FIR operates a linear convolution with the input signal. (the convolution results in a longer received vector but you can cut it down to what you believe to be the size of the original transmitted vector as the extra transmitted samples are irrelevant)

* Very low power noise was added.