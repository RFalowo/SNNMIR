import librosa
import torch
import torch.nn as nn
import tensorboard
from spikingjelly.clock_driven import neuron, functional, encoding, layer
from spikingjelly.activation_based import learning
from utils.filterbank import create_log_filter_bank, compute_power

# Load the audio file
def load_audio(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    return y, sr

# Beat Detection RSNN
class BeatDetectionRSNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, recurrent_size, output_size):
        super(BeatDetectionRSNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.lif1 = neuron.LIFNode()
        self.rlayer = layer.LIFRNNCell(hidden_size, recurrent_size)
        self.fc2 = torch.nn.Linear(recurrent_size, output_size)
        self.lif2 = neuron.LIFNode()

    def forward(self, x):
        seq_length = x.shape[1]
        h = None

        outputs = []
        for t in range(seq_length):
            x_t = x[:, t, :]
            x_t = self.fc1(x_t)
            x_t = self.lif1(x_t)
            h, x_t = self.rlayer(x_t, h)
            x_t = self.fc2(x_t)
            x_t = self.lif2(x_t)
            outputs.append(x_t)

        return torch.stack(outputs, dim=1)


class BeatDetectionRSNN2(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(BeatDetectionRSNN2, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size, bias=False)
        self.lif1 = neuron.IFNode()
        self.fc1_stdp = learning.STDPLearner(step_mode='s', synapse=self.fc1, sn=self.lif1,
                                             tau_pre=2.0, tau_post=2.0,
                                             f_pre=self.clamp_weights, f_post=self.clamp_weights)


        self.fc2 = nn.Linear(input_size, output_size, bias=False)
        self.lif2 = neuron.IFNode()
        self.fc2_stdp = learning.STDPLearner(step_mode='s', synapse=self.fc2, sn=self.lif2,
                                             tau_pre=2.0, tau_post=2.0,
                                             f_pre=self.clamp_weights, f_post=self.clamp_weights)

    def forward(self, x):
        h = None
        out_spike = []

        for t in range(x.size(1)):
            x_t = x[:, t, :]
            x_t = self.lif1(self.fc1(x_t))
            out = self.lif2(self.fc2(x_t))
            out_spike.append(out)

        return torch.stack(out_spike, dim=1)

    @staticmethod
    def clamp_weights(x):
        return torch.clamp(x, -1, 1.)

def main(audio_file):
    # Load audio and get sample rate
    audio, sr = load_audio(audio_file)

    # Extract features using filter bank
    filter_bank = create_log_filter_bank(16, 50, 10000, sr)
    power_features = compute_power(audio, filter_bank)

    # Convert the power features to spike trains
    threshold = 0.5 * max(power_features)
    spike_trains = encoding.bool2fire([int(p > threshold) for p in power_features])

    # Convert spike trains to tensor format suitable for the RSNN
    spike_trains = torch.FloatTensor(spike_trains).unsqueeze(0)

    # Initialize RSNN and pass spike trains through the network
    model = BeatDetectionRSNN(input_size=16, hidden_size=32, recurrent_size=64, output_size=1)
    output = model(spike_trains)

    # 'output' contains the spike data for beat detection.
    # Post-processing can be applied to 'output' to detect beats in the audio.

# if __name__ == "__main__":
#     audio_file = "path_to_your_audio_file.wav"
#     main(audio_file)

def train_with_stdp(model, data_loader, num_epochs=10, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.)

    for epoch in range(num_epochs):
        for spike_trains, target in data_loader:
            optimizer.zero_grad()

            # Forward pass
            output = model(spike_trains)

            # Apply STDP for fc1 and fc2 layers
            model.fc1_stdp.step(on_grad=True)
            model.fc2_stdp.step(on_grad=True)

            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]: Completed")

# Dummy data for the example
dummy_data = [torch.rand(1, 100, 16) for _ in range(1000)]
dummy_labels = [torch.rand(1, 100, 1) for _ in range(1000)]
train_data = list(zip(dummy_data, dummy_labels))
data_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)

# Initialize the model and training
model = BeatDetectionRSNN2(input_size=16, output_size=1)

train_with_stdp(model, data_loader)