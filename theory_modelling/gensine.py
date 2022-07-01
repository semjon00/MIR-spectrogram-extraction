# https://dsp.stackexchange.com/questions/53125/write-a-440-hz-sine-wave-to-wav-file-using-python-and-scipy

if __name__ == '__main__':
    pass

import numpy as np
from scipy.io import wavfile

sampleRate = 44100
frequency = 440
length = 0.1

t = np.linspace(0, length, round(sampleRate * length))  #  Produces a 5 second Audio-File
y = np.sin(frequency * 2 * np.pi * t)  #  Has frequency of 440Hz

wavfile.write('440.wav', sampleRate, y)
