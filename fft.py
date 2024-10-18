import cmath

def radix_rearrange(data):
    ''' Bit reversal for radix FFT --- takes any sized inputs --- operates inplace '''
    size = len(data)
    target = 0

    for position in range(0, size):
        if target > position:
            data[target], data[position] = data[position], data[target]
        
        mask = size >> 1
        
        while target & mask:
            target &= ~mask
            mask >>= 1

        target |= mask

    return data

def radix_fft(data, inverse = False):
    ''' Radix FFT algorithm --- works on powers of two only --- operates inplace '''
    sign_pi = complex(0, cmath.pi if inverse else -cmath.pi)
    size = len(data)
    half_size = size // 2
    
    radix_rearrange(data)

    twiddles = [None] * half_size
    for index in range(half_size):
        twiddles[index] = cmath.exp(sign_pi * index / half_size)

    step = 1
    while step < size:
        jump = step << 1

        twiddle = complex(1, 0)

        group = 0
        while group < step:

            pair = group
            while pair < size:
                
                match = pair + step
                product = twiddle * data[match]

                data[match] = data[pair] - product
                data[pair] += product

                pair += jump

            group += 1

            if group == step:
                continue

            twiddle = twiddles[group * half_size // step]

        step <<= 1

    return data

def fft(data, inverse = False):
    ''' Bluestein FFT algorithm --- works on general length inputs --- operates inplace '''
    sign_pi = complex(0, cmath.pi if inverse else -cmath.pi)
    size = len(data)

    larger_size = 1
    while larger_size >> 1 <= size:
        larger_size <<= 1

    chirp = [None] * size
    for index in range(size):
        chirp[index] = cmath.exp(sign_pi * index**2 / size)

    a = [complex(0,0)] * larger_size
    for index in range(size):
        a[index] = chirp[index] * data[index]

    b = [complex(0,0)] * larger_size
    b[0] = chirp[0]
    for index in range(1, size):
        b[index] = b[larger_size - index] = chirp[index].conjugate()

    radix_fft(a, inverse = False)
    radix_fft(b, inverse = False)

    for index in range(larger_size):
        a[index] *= b[index]

    radix_fft(a, inverse = True)
    
    for index in range(size):
        data[index] = a[index] * chirp[index] / larger_size

    return data

if __name__ == '__main__':

    data = [complex(0, i) for i in range(7)]

    print(data)
    fft(data)
    print(data)
