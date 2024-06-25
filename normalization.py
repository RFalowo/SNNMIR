import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from GTZANBEAT import GTZANDataset
import torchaudio.transforms as T


class TEBN(nn.Module):
    def __init__(self, num_features, max_timesteps, eps=1e-5, momentum=0.3, device='mps'):
        super(TEBN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.device = device

        # Initialize learnable parameters
        self.p = nn.Parameter(torch.ones(max_timesteps), requires_grad=True).to(self.device)
        self.gamma = nn.Parameter(torch.ones(num_features)).to(self.device)
        self.beta = nn.Parameter(torch.zeros(num_features)).to(self.device)

        # Initialize running mean and variance
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        # Get the number of time steps from the input shape
        num_timesteps = input.shape[0]

        # Reshape input to (num_timesteps, batch_size, num_features)
        shape = input.shape
        input = input.view(num_timesteps, shape[1], self.num_features)

        # Compute overall mean and variance across time and batch
        mean = input.mean(dim=(0, 1))
        var = input.var(dim=(0, 1), unbiased=False)

        # Update running mean and variance
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var

        # Normalize input
        norm_input = (input - mean.view(1, 1, self.num_features)) / torch.sqrt(
            var.view(1, 1, self.num_features) + self.eps)

        # Apply time-specific scaling and shifting
        scaled_input = norm_input * self.gamma.view(1, 1, self.num_features) * self.p[:num_timesteps].view(num_timesteps, 1, 1)
        shifted_input = scaled_input + self.beta.view(1, 1, self.num_features)

        # Reshape output to original shape
        output = shifted_input.view(shape)

        return output


def compute_dataset_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    mean_mfcc = torch.zeros(40)  # Initialize mean tensor for MFCC
    mean_mfcc = mean_mfcc.squeeze()  # Remove batch dimension
    std_mfcc = torch.zeros_like(mean_mfcc)  # Initialize std tensor for MFCC
    nb_samples = 0

    for idx, (mfcc, beat) in enumerate(loader):
        # Accumulate mean and std for MFCC
        print(mfcc.shape)
        mfcc = mfcc.squeeze()# Remove batch dimension
        print(mfcc.shape)
        mean_mfcc += mfcc.mean(1)  # Compute mean along time dimension
        std_mfcc += mfcc.std(1)  # Compute std along time dimension


        nb_samples += 1

    mean_mfcc /= nb_samples
    std_mfcc /= nb_samples


    return mean_mfcc, std_mfcc

def save_normalization_values(mean_mfcc, std_mfcc, mean_beat, std_beat, filename):
    with open(filename, 'w') as f:
        f.write(f"Mean MFCC: {mean_mfcc.tolist()}\n")
        f.write(f"Std MFCC: {std_mfcc.tolist()}\n")


# audio_path = 'data/Data/genres_original'
# beat_path = 'data/gtzan_tempo_beat/beats'
# # Define your transform here, it's important to ensure that the transform is consistent with how you plan to process the data during training
# transform = T.MFCC( n_mfcc=40, melkwargs={'hop_length': 512})
#
# dataset = GTZANDataset(audio_dir=audio_path, beat_dir=beat_path, transform=transform)
# mean_mfcc, std_mfcc= compute_dataset_mean_std(dataset)
# save_normalization_values(mean_mfcc, std_mfcc, 'normalization_values.txt')
# print(f"Mean MFCC: {mean_mfcc}, Std MFCC: {std_mfcc}")

