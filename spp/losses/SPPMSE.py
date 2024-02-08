import torch
from torch import nn

from .. import utils

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)


class SPPMSE(nn.Module):
    """
    compute oracle ML-like SPP, and use that as the target
    """

    def __init__(
        self,
        frame_length: int = 512,
        shift_length: int = 256,
        xi_h1: float = 31.622776601683793,  # 15 dB
        p_h0: float = 0.5,
        p_h1: float = 0.5,
        fs: int = 16000,
        smoothing_noise_time: float = 0.072,  # from [1] T. Gerkmann and R. Hendriks, “Unbiased MMSE-Based Noise Power Estimation With Low Complexity and Low Tracking Delay,” IEEE Trans. Audio, Speech and Lang. Proc., vol. 20, no. 4, pp. 1383–1393, May 2012, doi: 10.1109/TASL.2011.2180896.
        **kwargs,
    ):
        super().__init__()
        self.xi_h1 = xi_h1
        self.p_h0 = p_h0
        self.p_h1 = p_h1
        self.alpha = utils.time_to_smoothing_constant(
            smoothing_noise_time, shift_length=shift_length, fs=fs
        )
        self.kwargs = kwargs

        self.stft = utils.STFTTorch(
            frame_length=frame_length,
            overlap_length=frame_length - shift_length,
            window=torch.hann_window,
            sqrt=True,
        )

    def forward(self, outputs, extras, meta):
        # extract required signals
        try:
            stft_noise = self.stft.get_stft(extras["interference"]).squeeze(1)
        except KeyError:
            stft_noise = self.stft.get_stft(extras["input"] - extras["clean"]).squeeze(
                1
            )
        stft_noisy = self.stft.get_stft(extras["input"]).squeeze(1)
        spp_estimate = outputs["spp_estimate"].squeeze(1)
        spp_oracle = self.estimate_spp(stft_noise, stft_noisy)

        return {
            "loss": ((spp_estimate - spp_oracle) ** 2).mean(),
            "spp_oracle": spp_oracle,
            "spp_estimate": spp_estimate,
        }

    def estimate_spp(
        self,
        stft_noise: torch.Tensor,
        stft_noisy: torch.Tensor,
    ) -> torch.Tensor:
        # recursively compute noise PSD estimate
        num_frames = stft_noise.shape[-1]
        var_noise = stft_noise.new_empty(stft_noise.shape, dtype=torch.float)
        var_noise[..., 0] = stft_noise[..., 0].abs().pow(2)
        for frame in torch.arange(1, num_frames):
            var_noise[..., frame] = self.alpha * var_noise[..., frame - 1] + (
                1.0 - self.alpha
            ) * stft_noise[..., frame].abs().pow(2)
        return self.get_a_posteriori_spp(stft_noisy, var_noise)

    def get_a_posteriori_spp(self, stft_noisy, var_noise):
        a_posteriori_spp = 1.0 / (
            1.0
            + self.p_h0
            / self.p_h1
            * (1.0 + self.xi_h1)
            * torch.exp(
                -stft_noisy.abs().pow(2)
                / (var_noise + EPS)
                * self.xi_h1
                / (1.0 + self.xi_h1)
            )
        )
        return a_posteriori_spp
