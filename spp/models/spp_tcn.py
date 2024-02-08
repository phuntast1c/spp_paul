import math
from typing import Optional, Union

import matplotlib
import torch

from .. import building_blocks as bb
from .. import losses, utils
from . import BaseLitModel

matplotlib.use("agg")

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class SPPEstimator(BaseLitModel):
    def __init__(
        self,
        learning_rate: float = 0.0003,
        batch_size: int = 8,
        metrics_test: Union[
            tuple, str
        ] = "PESQWB,PESQNB,PESQNBRAW,STOI,ESTOI,DNSMOS,SISDR",
        metrics_val: Union[tuple, str] = "",
        frame_length: int = 128,
        shift_length: int = 32,
        window_type: str = "hann",
        layer: int = 4,
        stack: int = 2,
        kernel: int = 3,
        hidden_dim: Optional[int] = None,
        fs: int = 16000,
        use_batchnorm: bool = True,
        use_log: bool = True,
        xi_h1: float = 31.622776601683793,  # 15 dB
        p_h0: float = 0.5,
        p_h1: float = 0.5,
        smoothing_noise_time: float = 0.072,
        my_lr_scheduler: str = "ReduceLROnPlateau",
    ):
        super().__init__(
            lr=learning_rate,
            batch_size=batch_size,
            metrics_test=metrics_test,
            metrics_val=metrics_val,
            model_name="SPPEstimator",
            my_lr_scheduler=my_lr_scheduler,
        )
        self.loss = losses.SPPMSE(
            frame_length=frame_length,
            shift_length=shift_length,
            xi_h1=xi_h1,
            p_h0=p_h0,
            p_h1=p_h1,
            fs=fs,
            smoothing_noise_time=smoothing_noise_time,
        )

        self.frame_length = frame_length
        self.shift_length = shift_length
        self.window_type = window_type
        self.layer = layer
        self.stack = stack
        self.kernel = kernel
        self.frequency_bins = int(self.frame_length / 2) + 1
        self.bn_dim = self.frequency_bins // 4 if hidden_dim is None else hidden_dim
        self.fs = fs
        self.use_batchnorm = use_batchnorm
        self.use_log = use_log

        self.output_size = self.frequency_bins

        self.estimator = bb.TCNEstimator(
            input_dim=3 * self.frequency_bins,  # mag, cos, sin
            output_dim=self.output_size,
            BN_dim=self.bn_dim,
            hidden_dim=4 * self.bn_dim,
            layer=int(self.layer),
            stack=int(self.stack),
            kernel=int(self.kernel),
        )

        self.stft = utils.STFTTorch(
            frame_length=self.frame_length,
            overlap_length=self.frame_length - self.shift_length,
            window=self.window_type,
            sqrt=self.window_type == "hann",
        )

        if self.use_batchnorm:
            self.batchnorm1d_noisy = (
                torch.nn.BatchNorm1d(  # used for feature normalization
                    num_features=self.frequency_bins,
                )
            )

        self.num_params = self.count_parameters()
        self.save_hyperparameters()

    def forward(self, batch):
        """
        # dimensions convention:
        batch_size x channels x F x T x filter_length x filter_length
        """
        noisy = batch["input"]
        noisy = torch.stack([self.stft.get_stft(x) for x in noisy])
        noisy = noisy.squeeze(1)

        try:
            self.testing = self.trainer.testing
        except RuntimeError:
            self.testing = True

        # use (log) magnitude and phase spectra
        noisy_mag = noisy.abs() + EPS
        if self.use_log:
            noisy_mag = noisy_mag.log10()
        noisy_angle = noisy.angle()
        noisy_phase_cos = noisy_angle.cos()
        noisy_phase_sin = noisy_angle.sin()

        if self.use_batchnorm:
            noisy_mag = self.batchnorm1d_noisy(noisy_mag)

        features_cat = torch.cat([noisy_mag, noisy_phase_cos, noisy_phase_sin], dim=1)

        spp_estimate = torch.sigmoid(self.estimator(features_cat))

        return {
            "spp_estimate": spp_estimate,
        }
