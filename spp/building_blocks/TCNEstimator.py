import torch
from torch import nn

from . import DepthConv1d, cLN


class TCNEstimator(nn.Module):
    """
    small modification of original code of TCN from Conv-TasNet for STFT-domain parameter estimation
    https://github.com/naplab/Conv-TasNet
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        BN_dim,
        hidden_dim,
        layer=8,
        stack=3,
        kernel=3,
        skip=True,
        causal=True,
        dilated=True,
        use_encoder=False,
        sr=16000,
        win=2,
        stride=None,
    ):
        super().__init__()

        # input is a sequence of features of shape (B, N, L)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.BN_dim = BN_dim
        self.hidden_dim = hidden_dim

        self.use_encoder = use_encoder
        if self.use_encoder:
            self.enc_dim = input_dim
            self.win = int(sr * win / 1000)  # window size in ms
            self.stride = self.win // 2 if stride is None else stride
            self.encoder = nn.Conv1d(
                1, self.enc_dim, self.win, bias=False, stride=self.stride
            )

        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, BN_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for s in range(stack):
            for i in range(layer):
                if self.dilated:
                    self.TCN.append(
                        DepthConv1d(
                            BN_dim,
                            hidden_dim,
                            kernel,
                            dilation=2**i,
                            padding=2**i,
                            skip=skip,
                            causal=causal,
                        )
                    )
                else:
                    self.TCN.append(
                        DepthConv1d(
                            BN_dim,
                            hidden_dim,
                            kernel,
                            dilation=1,
                            padding=1,
                            skip=skip,
                            causal=causal,
                        )
                    )
                if i == 0 and s == 0:
                    self.receptive_field += kernel
                else:
                    self.receptive_field += (
                        (kernel - 1) * 2 ** i if self.dilated else (kernel - 1)
                    )
        print("receptive field: {:d} time steps".format(self.receptive_field))

        self.output = nn.Conv1d(BN_dim, output_dim, 1)

        self.skip = skip

    def pad_signal(self, signal):
        # inp is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if signal.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")

        if signal.dim() == 2:
            signal = signal.unsqueeze(1)
        batch_size = signal.size(0)
        nsample = signal.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = torch.zeros((batch_size, 1, rest), requires_grad=True).type(
                signal.type()
            )
            signal = torch.cat([signal, pad], 2)

        pad_aux = torch.zeros((batch_size, 1, self.stride), requires_grad=True).type(
            signal.type()
        )
        signal = torch.cat([pad_aux, signal, pad_aux], 2)

        return signal, rest

    def forward(self, inp):
        if self.use_encoder:
            raise DeprecationWarning("use_encoder is deprecated")
            # optional: encoding wave inp: (B, L)
            output, _ = self.pad_signal(inp)
            output = self.encoder(output)  # B, N, L
        else:
            output = inp

        # inp shape: (B, N, L)
        # normalization
        output = self.BN(self.LN(output))

        # pass to TCN
        if self.skip:
            skip_connection = 0.0
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection += skip
        else:
            for i in range(len(self.TCN)):
                residual, _ = self.TCN[i](output)
                output = output + residual

        # output layer
        output = self.output(skip_connection) if self.skip else self.output(output)
        return output
