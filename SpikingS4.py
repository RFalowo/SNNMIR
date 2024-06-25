import torch
import torch.nn as nn
import spikingjelly as sj
from spikingjelly.clock_driven import surrogate, neuron, functional
from spikingjelly.activation_based import layer

import torch
import torch.nn as nn

class StructuredStateSpace(nn.Module):
    def __init__(self, sub_module: nn.Module, in_features: int, out_features: int, f_hippo=True, bias=True, step_mode='s'):
        super(StructuredStateSpace, self).__init__()
        self.sub_module = sub_module
        self.in_features = in_features
        self.out_features = out_features
        self.step_mode = step_mode

        self.A = layer.Linear(out_features, out_features, bias=False)
        self.B = layer.Linear(in_features, out_features, bias=bias)
        self.C = layer.Linear(out_features, in_features, bias=False)

        if f_hippo:
            # Initialize A with the HIPPO matrix
            self.A.weight.data = torch.from_numpy(hippo_matrix(out_features)).float()

        self.recurrent_container = layer.LinearRecurrentContainer(
            sub_module=nn.Sequential(self.A, self.C),
            in_features=in_features + out_features,
            out_features=in_features,
            bias=bias,
            step_mode=step_mode
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.transpose(0, 1)  # (seq_len, batch_size, in_features)

        outputs = []
        for t in range(seq_len):
            input_t = x[t] if t == 0 else torch.cat((x[t], outputs[-1]), dim=-1)
            output_t = self.recurrent_container(input_t)
            outputs.append(output_t)

        outputs = torch.stack(outputs, dim=0).transpose(0, 1)  # (batch_size, seq_len, in_features)
        return outputs

def hippo_matrix(hidden_size):
    # Implementation of the HIPPO matrix from the S4 paper
    import numpy as np
    from scipy.stats import ortho_group

    def f(x):
        return np.arctan(x)

    dim = hidden_size
    Q = ortho_group.rvs(dim)
    Q_cost = Q @ f(Q.T)
    return Q_cost

class SpikeS4Layer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, time_constant=2.0):
        super(SpikeS4Layer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.time_constant = time_constant

        self.s4_kernel = StructuredStateSpace(in_features=in_channels, out_features=hidden_size, f_hippo=True)
        self.emission = nn.Linear(hidden_size, out_channels)
        self.lif_node = neuron.LIFNode(tau=time_constant)
        self.decoder = nn.Linear(out_channels, out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        batch_size, time_steps, _ = x.size()

        # Flatten the input
        x = x.view(batch_size * time_steps, self.in_channels)

        # Pass through S4 kernel
        s4_output = self.s4_kernel(x)

        # Emission layer
        emission_output = self.emission(s4_output)

        # LIF node
        spike_output = self.lif_node(emission_output)

        # Decoder
        decoder_output = self.decoder(spike_output)

        # Shortcut connection
        shortcut_output = self.shortcut(x)
        output = decoder_output + shortcut_output

        # Reshape the output
        output = output.view(batch_size, time_steps, self.out_channels)

        return output

class SpikeS4Model(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_kernels, num_layers, time_constant=2.0):
        super(SpikeS4Model, self).__init__()
        self.encoder = nn.Linear(in_channels, hidden_size)
        self.spiking_s4_layers = nn.ModuleList([
            SpikeS4Layer(hidden_size, hidden_size, hidden_size, num_kernels, time_constant)
            for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
        encoded = self.encoder(x)
        for layer in self.spiking_s4_layers:
            encoded = layer(encoded)
        output = self.decoder(encoded)
        return output