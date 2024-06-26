import torch
import torch.nn as nn
import spikingjelly as sj
from spikingjelly.clock_driven import surrogate, neuron, functional, layer

import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, neuron, functional, layer


import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, neuron, functional, layer

def parallel_scan(inputs, func, dim=0):
    outputs = torch.zeros_like(inputs)
    outputs[0] = inputs[0]
    for i in range(1, inputs.size(dim)):
        outputs[i] = func(outputs[i-1], inputs[i])
    return outputs

class S4Layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, f_hippo=True, bias=True, neuron_type='none'):
        super(S4Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.neuron_type = neuron_type

        # Initialize layers
        self.A = layer.Linear(out_features, out_features, bias=False)
        self.B = layer.Linear(in_features, out_features, bias=bias)
        self.C = layer.Linear(out_features, in_features, bias=False)

        if f_hippo:
            self.A.weight.data = torch.from_numpy(self.hippo_matrix(out_features)).float()

        self.recurrent_container = layer.LinearRecurrentContainer(
            sub_module=nn.Sequential(self.A, self.C),
            in_features=in_features + out_features,
            out_features=in_features,
            bias=bias
        )

        # Initialize neuron layer if specified
        if neuron_type == 'lif':
            self.neuron_layer = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
            self.neuron_layer.tau = nn.Parameter(self.neuron_layer.tau)
        elif neuron_type == 'lif_sr':
            self.neuron_layer = neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), spike_return=True)
            self.neuron_layer.tau = nn.Parameter(self.neuron_layer.tau)
        else:
            self.neuron_layer = None

    def reset(self):
        functional.reset_net(self)

    def forward_step(self, prev_output, current_input):
        input_combined = torch.cat((current_input, prev_output), dim=-1)
        output = self.recurrent_container(input_combined)
        if self.neuron_layer:
            output = self.neuron_layer(output)
        return output

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.transpose(0, 1)  # (seq_len, batch_size, in_features)

        self.reset()
        initial_output = torch.zeros(batch_size, self.out_features, device=x.device)
        outputs = parallel_scan(x, self.forward_step, dim=0)
        outputs = outputs.transpose(0, 1)  # (batch_size, seq_len, in_features)
        return outputs

    @staticmethod
    def hippo_matrix(hidden_size):
        import numpy as np
        from scipy.stats import ortho_group
        def f(x):
            return np.arctan(x)
        dim = hidden_size
        Q = ortho_group.rvs(dim)
        D = np.diag(f(np.linspace(0, np.pi, dim)))
        A = Q @ D @ Q.T
        return A
