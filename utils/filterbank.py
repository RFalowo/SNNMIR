import librosa
import numpy as np
import scipy.signal


def create_log_filter_bank(num_bands, low_freq, high_freq, sr, n_fft):
    # Create log-spaced frequencies
    start_freq = np.log10(low_freq)
    end_freq = np.log10(high_freq)
    center_freqs = np.logspace(start_freq, end_freq, num_bands)

    filter_bank = []
    for f in center_freqs:
        # Design band-pass filters around each center frequency
        nyquist = 0.5 * sr
        low = f - (f / 2 ** (1 / 2))
        high = f + (f / 2 ** (1 / 2))
        b, a = scipy.signal.butter(1, [low / nyquist, high / nyquist], btype='band')
        filter_bank.append((b, a))

    return filter_bank


def compute_power(audio, filter_bank):
    power_values = []
    for b, a in filter_bank:
        # Filter the audio signal
        y = scipy.signal.lfilter(b, a, audio)
        # Compute the power
        power = np.mean(y ** 2)
        power_values.append(power)
    return power_values


# # Load the audio file
# audio_file = 'path_to_your_audio_file.wav'
# y, sr = librosa.load(audio_file, sr=None)
#
# # Create the filter bank
# num_bands = 16
# low_freq = 50
# high_freq = 10000
# n_fft = 2048
# filter_bank = create_log_filter_bank(num_bands, low_freq, high_freq, sr, n_fft)
#
# # Compute the power for each band
# power_values = compute_power(y, filter_bank)

# print(power_values)
