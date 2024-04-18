import torchaudio
from lyon.calc import LyonCalc
import matplotlib.pyplot as plt
import numpy as np


def lyon_cochleagram(waveform, sample_rate, decimation_factor=32):
    waveform = np.float64(waveform).squeeze()
    calc = LyonCalc()

    # Now, signal.size is a multiple of decimation_factor
    coch = calc.lyon_passive_ear(waveform, sample_rate, decimation_factor)
    return coch

#load the audio file
# waveform, sample_rate = torchaudio.load('data/Data/genres_original/blues/blues.00000.wav')
# #calculate the cochleagram
# coch = lyon_cochleagram(np.float64(waveform).squeeze(), sample_rate)
# #plot the cochleagram
# plt.imshow(coch, aspect='auto', origin='lower')
# plt.show()

# foo = 1
