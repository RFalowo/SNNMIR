import torch
from torch.utils.data import DataLoader
from GTZANBEAT import GTZANDataset
import torchaudio.transforms as T

def compute_dataset_min_max(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    min_mfcc = None
    max_mfcc = None

    for _, (mfcc, _) in enumerate(loader):
        mfcc = mfcc.squeeze()  # Remove batch dimension
        if min_mfcc is None or max_mfcc is None:
            min_mfcc = mfcc.min(1).values
            max_mfcc = mfcc.max(1).values
        else:
            min_mfcc = torch.min(min_mfcc, mfcc.min(1).values)
            max_mfcc = torch.max(max_mfcc, mfcc.max(1).values)

    return min_mfcc, max_mfcc

def save_normalization_values(min_mfcc, max_mfcc, filename):
    with open(filename, 'w') as f:
        f.write(f"Min MFCC: {min_mfcc.tolist()}\n")
        f.write(f"Max MFCC: {max_mfcc.tolist()}\n")

audio_path = 'data/Data/genres_original'
beat_path = 'data/gtzan_tempo_beat/beats'
transform = T.MFCC(n_mfcc=40, melkwargs={'hop_length': 512})

dataset = GTZANDataset(audio_dir=audio_path, beat_dir=beat_path, transform=transform)
min_mfcc, max_mfcc = compute_dataset_min_max(dataset)
save_normalization_values(min_mfcc, max_mfcc, 'normalizationMM_values.txt')
print(f"Min MFCC: {min_mfcc}, Max MFCC: {max_mfcc}")
