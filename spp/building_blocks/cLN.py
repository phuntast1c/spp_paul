import torch
from torch import nn


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super().__init__()

        self.eps = eps
        self.gain = nn.Parameter(torch.ones((1, dimension, 1), requires_grad=trainable))
        self.bias = nn.Parameter(
            torch.zeros((1, dimension, 1), requires_grad=trainable)
        )

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        channel = input.size(1)
        time_step = input.size(2)

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = torch.arange(
            channel, channel * (time_step + 1), channel, device=input.device
        )
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(
            2
        )  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(
            x.type()
        )
