import torch
import torch.nn as nn
from spikingjelly.clock_driven import surrogate, neuron, functional, layer
import numpy as np
from scipy.stats import ortho_group


class SpikingS4Layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int, L: int, f_hippo=True, bias=True,
                 neuron_type='lif', delta=0.1):
        super(SpikingS4Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.L = L
        self.neuron_type = neuron_type
        self.delta = delta

        # Initialize S4 kernels
        self.s4_kernels = nn.ModuleList([
            StructuredStateSpace(in_features, hidden_size, f_hippo, bias, delta)
            for _ in range(L)
        ])

        # Initialize emission layer
        self.emission_layer = nn.ModuleList([
            nn.Linear(hidden_size, 1, bias=bias)
            for _ in range(L)
        ])

        # Initialize neuron layer if specified
        if neuron_type == 'lif':
            self.neuron_layer = neuron.LIFNode(surrogate_function=surrogate.Sigmoid())
        elif neuron_type == 'lif_sr':
            self.neuron_layer = neuron.LIFNode(surrogate_function=surrogate.Sigmoid(), spike_return=True)
        else:
            self.neuron_layer = None

        # Initialize final linear decoder
        self.linear_decoder = nn.Linear(L, out_features)

    def reset(self):
        # Reset neuron layer if it exists
        if self.neuron_layer:
            functional.reset_net(self.neuron_layer)

        # Reset each S4 kernel
        for s4_kernel in self.s4_kernels:
            s4_kernel.reset()

    def forward(self, x):
        seq_len, batch_size,  _ = x.size()
        self.reset()

        outputs = []
        for t in range(seq_len):
            input_t = x[t]
            hidden_states = []
            for i in range(self.L):
                kernel_output = self.s4_kernels[i](input_t)
                emission_output = self.emission_layer[i](kernel_output)
                hidden_states.append(emission_output)
            hidden_states = torch.stack(hidden_states, dim=-1).squeeze(-1)

            if self.neuron_layer:
                hidden_states = self.neuron_layer(hidden_states)

            outputs.append(hidden_states)

        outputs = torch.stack(outputs, dim=0).transpose(0, 1)  # (batch_size, seq_len, L)
        outputs = self.linear_decoder(outputs)  # (batch_size, seq_len, out_features)
        return outputs


class StructuredStateSpace(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, f_hippo=True, bias=True, delta=0.1):
        super(StructuredStateSpace, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.delta = delta

        # Initialize layers
        self.A_cont = self.hippo_matrix(hidden_size)
        self.B_cont = nn.Parameter(torch.randn(hidden_size, in_features))

        self.A, self.B = self.discretize(self.A_cont, self.B_cont, delta)

        self.A = nn.Parameter(self.A)
        self.B = nn.Parameter(self.B)
        self.C = nn.Parameter(torch.randn(in_features, hidden_size))

        self.K = self.compute_cauchy_kernel(self.A, self.B, self.C)

    def discretize(self, A, B, delta):
        I = torch.eye(A.shape[0])
        Ad = torch.inverse(I - delta / 2 * A) @ (I + delta / 2 * A)
        Bd = torch.inverse(I - delta / 2 * A) @ (delta * B)
        return Ad, Bd

    def compute_cauchy_kernel(self, A, B, C):
        # Assuming A, B, C are already discrete
        kernel_size = 10  # Example kernel size, adjust as necessary
        K = []
        Ak = torch.eye(A.size(0))
        for i in range(kernel_size):
            if i > 0:
                Ak = Ak @ A
            K.append(C @ Ak @ B)
        K = torch.stack(K, dim=0)
        return K

    def reset(self):
        # Any specific reset logic for StructuredStateSpace can be implemented here
        pass

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        device = x.device
        x = x.transpose(0, 1)  # (seq_len, batch_size, in_features)

        outputs = []
        for t in range(seq_len):
            if t < self.K.size(0):
                k_slice = self.K[:t + 1].to(device)
            else:
                k_slice = self.K.to(device)
            inputs = x[:t + 1]
            input_rev = torch.flip(inputs, [0]).to(device)  # Reverse the input sequence
            k_slice_rev = torch.flip(k_slice, [0]).to(device)  # Reverse the kernel slice

            # Ensure tensors are 3-dimensional
            if input_rev.dim() == 2:
                input_rev = input_rev.unsqueeze(-1)
            if k_slice_rev.dim() == 2:
                k_slice_rev = k_slice_rev.unsqueeze(-1)

            # Check and adjust dimensions to match
            if input_rev.size(1) != k_slice_rev.size(1):
                input_rev = input_rev.expand_as(k_slice_rev)

            y_t = torch.einsum('ijk,jlk->il', k_slice_rev, input_rev)
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=0).transpose(0, 1)  # (batch_size, seq_len, in_features)
        return outputs

    @staticmethod
    def hippo_matrix(hidden_size):
        def f(x):
            return np.arctan(x)

        dim = hidden_size
        Q = ortho_group.rvs(dim)
        D = np.diag(f(np.linspace(0, np.pi, dim)))
        A = Q @ D @ Q.T
        return torch.from_numpy(A).float()