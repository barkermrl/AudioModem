import ldpc_jossy.py.ldpc as ldpc
import numpy as np
def get_code():
    ''''Standard Code Definition:'''
    STANDARD = '802.16'
    Z = 64
    RATE = '1/2'
    c = ldpc.code(standard = STANDARD, z = Z, rate = RATE)
    return c
def encode(source_bits, c):
    ''''Length of the source bits must be divisible by the total code length,
        if not then pad the source bits by 0s. This should
        not be a problem because the number of bits is included in the meta data'''
    if len(source_bits)%c.K != 0:
        source_bits = np.pad(source_bits, (0, c.K - len(source_bits)%c.K))
    bits = np.split(source_bits, len(source_bits)//c.K)
    codewords = np.zeros(shape = (len(source_bits)//c.K, c.N))
    for i, row in enumerate(bits):
        codewords[i,:] = c.encode(row)
    return codewords.flatten()
def decode(lrr,c):
    app, it = c.decode(llr)
    u_hat = (app<0).toint()
    return u_hat
def get_llr(constelation, sigma2s):
    assert len(constelation) == len(sigma2s)
    llr = []
    for x_hat, sigma2 in zip(constelation, sigma2s):
        llr.append(x_hat.imag/sigma2)
        llr.append(x_hat.real/sigma2)
    return llr
u = np.random.choice([0,1], 1000)
c = get_code()
x = encode(u, c)
print(x)