
def as_bit_array(x: int, bit_count=8):
    bitarray = [0] * bit_count
    for i, bit in enumerate(bin(x)[2:].zfill(8)):
        bitarray[i] = int(bit)
    return bitarray
