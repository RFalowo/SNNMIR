import os
import torch
import torchaudio
import torch.nn.functional as F
from lyon.calc import LyonCalc
import numpy as np

class GTZANDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, beat_dir, transform=None, gaussian_width=5, normalization_file=None):
        self.audiofiles = []
        self.beatfiles = []
        print(f"Transform argument: {transform}")
        self.transform = transform
        self.gaussian_width = gaussian_width
        self.normalization_file = normalization_file
        self.min_mfcc, self.max_mfcc = self.load_normalization_values()  # Load normalization values
        # Iterate through each genre directory in audio_dir
        for genre in os.listdir(audio_dir):
            genre_path = os.path.join(audio_dir, genre)
            if not os.path.isdir(genre_path):
                continue

            # For each audio file in the genre directory
            for audio_file in os.listdir(genre_path):
                if audio_file.endswith('.wav'):
                    self.audiofiles.append(os.path.join(genre_path, audio_file))

                    # Construct beat file path and add to beatfiles list
                    audio_filename_parts = audio_file.split('.')
                    beat_filename = "gtzan_" + '_'.join(audio_filename_parts[0:2])  + '.beats'
                    self.beatfiles.append(os.path.join(beat_dir, beat_filename))

    def __len__(self):
        return len(self.audiofiles)

    def __getitem__(self, idx):
        audio_path = self.audiofiles[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        if self.transform:
            transformed_data = self.transform(waveform, sample_rate)

        # if self.mean_mfcc is not None and self.std_mfcc is not None:
            # Reshape mean and std for broadcasting, assuming mfcc shape is [batch, features, time]
            # mean_mfcc = self.mean_mfcc.reshape(1, -1, 1)
            # std_mfcc = self.std_mfcc.reshape(1, -1, 1)
            # mfcc = (mfcc - mean_mfcc) / (std_mfcc + 1e-6)  # Adding a small epsilon to avoid division by zero
        if self.transform == 'mfcc':
            if self.min_mfcc is not None and self.max_mfcc is not None:
                # Ensure min and max values are properly broadcasted over the MFCC dimensions
                transformed_data = (transformed_data - self.min_mfcc.reshape(1, -1, 1)) / (
                            self.max_mfcc.reshape(1, -1, 1) - self.min_mfcc.reshape(1, -1, 1) + 1e-6)

        item_length = transformed_data.shape[0]

        beat_path = self.beatfiles[idx]
        beat_tensor = torch.zeros(item_length)

        with open(beat_path, 'r') as f:
            beats = [float(line.split('\t')[0].strip()) for line in f.readlines()]

        hop_length = 512  # Assuming hop length used in MFCC; adjust as needed
        beat_frames = [int(round(beat_time * sample_rate / hop_length)) for beat_time in beats]

        gaussian_width = self.gaussian_width
        for beat_frame in beat_frames:
            if 0 <= beat_frame < item_length:
                start = max(0, beat_frame - gaussian_width)
                end = min(item_length, beat_frame + gaussian_width + 1)
                x = torch.arange(start, end) - beat_frame
                beat_tensor[start:end] = torch.exp(-0.5 * (x / gaussian_width) ** 2)

        audio = torch.tensor(transformed_data, dtype=torch.float32).permute(1, 0)
        audio = F.normalize(audio, dim=0)
        return audio, beat_tensor

    def load_normalization_values(self):
        if self.normalization_file is None:
            return None, None  # Default values if no file is specified
        with open(self.normalization_file, 'r') as f:
            lines = f.readlines()
            min_mfcc = torch.tensor(eval(lines[0].split(': ')[1]))
            max_mfcc = torch.tensor(eval(lines[1].split(': ')[1]))
        return min_mfcc, max_mfcc

#lyon cochleagram tranform class
class LyconCoch(object):
    def __init__(self, sample_rate, decimation_factor=32):
        self.sample_rate = sample_rate
        self.decimation_factor = decimation_factor
        self.calc = LyonCalc()

    def __call__(self, waveform):
        waveform = np.float64(waveform).squeeze()
        coch = self.calc.lyon_passive_ear(waveform, self.sample_rate, self.decimation_factor)
        return coch

