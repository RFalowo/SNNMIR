import torch
from spikingjelly.activation_based import learning
from spikingjelly.activation_based.learning import stdp_conv1d_single_step



def f_weight(x):
    return torch.clamp(x, -1, 1.)

class CustomSTDPLearner(learning.STDPLearner):
    def step(self, on_grad: bool = True, scale: float = 1.):
        length = self.in_spike_monitor.records.__len__()
        delta_w = None

        # Reset trace_pre and trace_post at the beginning of each new data point
        self.trace_pre = torch.zeros_like(self.in_spike_monitor.records[0])
        self.trace_post = torch.zeros_like(self.out_spike_monitor.records[0])

        if self.step_mode == 's':
            if isinstance(self.synapse, torch.nn.Conv1d):
                stdp_f = stdp_conv1d_single_step
            else:
                raise NotImplementedError(self.synapse)
        else:
            raise ValueError(self.step_mode)



        for _ in range(length):
            in_spike = self.in_spike_monitor.records.pop(0)
            out_spike = self.out_spike_monitor.records.pop(0)
            self.trace_pre, self.trace_post, dw = stdp_f(
                self.synapse, in_spike, out_spike,
                self.trace_pre, self.trace_post,
                self.tau_pre, self.tau_post,
                self.f_pre, self.f_post
            )
            if scale != 1.:
                dw *= scale
            delta_w = dw if (delta_w is None) else (delta_w + dw)

        if on_grad:
            if self.synapse.weight.grad is None:
                self.synapse.weight.grad = -delta_w
            else:
                self.synapse.weight.grad -= delta_w
        else:
            return delta_w