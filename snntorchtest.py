import torch
import torch.nn as nn
import snntorch as snn

alpha = 0.9
beta = 0.85

num_steps = 100
num_inputs = 50
num_hidden = 20

class Net(nn.module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_inputs, num_hidden)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []
