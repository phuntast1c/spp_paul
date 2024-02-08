import datetime
import math
import os
from dataclasses import dataclass

import human_readable_ids as hri
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

EPS = torch.as_tensor(torch.finfo(torch.get_default_dtype()).eps)
PI = math.pi


class STFTTorch:
    """
    class used to simplify handling of STFT & iSTFT
    """

    def __init__(
        self,
        frame_length=64,
        overlap_length=48,
        window=torch.hann_window,
        sqrt=True,
        normalized: bool = False,
        center: bool = True,
        fft_length=None,
        fft_length_synth=None,
        synthesis_window=None,
        use_my_istft: bool = False,
    ):
        self.frame_length = frame_length
        if fft_length is None:
            self.fft_length = frame_length
        else:
            self.fft_length = fft_length

        if fft_length_synth is None:
            self.fft_length_synth = fft_length
        else:
            self.fft_length_synth = fft_length_synth

        self.num_bins = int((self.fft_length / 2) + 1)
        self.overlap_length = overlap_length
        self.shift_length = self.frame_length - self.overlap_length
        self.sqrt = sqrt
        self.normalized = normalized
        self.center = center
        self.use_my_istft = use_my_istft

        if isinstance(window, str):
            if window == "hann":
                window = torch.hann_window
            elif window == "hamming":
                window = torch.hamming_window
            elif window == "bartlett":
                window = torch.bartlett_window
            elif window == "blackman":
                window = torch.blackman_window
            else:
                raise ValueError("unknown window type!")
            self.window = window(
                self.frame_length,
                periodic=True,
                dtype=torch.get_default_dtype(),
            )
        elif callable(window):
            self.window = window(
                self.frame_length,
                periodic=True,
                dtype=torch.get_default_dtype(),
            )
        elif type(window) is torch.Tensor:
            self.window = window
        else:
            raise NotImplementedError()

        if self.sqrt:
            self.window = self.window.sqrt()

        if synthesis_window is None:
            self.synthesis_window = self.window
        else:
            self.synthesis_window = synthesis_window

    def get_stft(self, wave):
        if self.window.device != wave.device:
            # move to device
            self.window = self.window.to(device=wave.device)
        shape_orig = wave.shape
        if wave.ndim > 2:  # reshape required
            wave = wave.reshape(-1, shape_orig[-1])
        stft = torch.stft(
            wave,
            window=self.window,
            n_fft=self.fft_length,
            hop_length=self.shift_length,
            win_length=self.frame_length,
            normalized=self.normalized,
            center=self.center,
            pad_mode="constant",
            return_complex=True,
        )
        return stft.reshape((*shape_orig[:-1], *stft.shape[-2:]))

    def get_istft(self, stft, length=None):
        if self.synthesis_window.device != stft.device:
            # move to device
            self.synthesis_window = self.synthesis_window.to(stft.device)

        if stft.ndim == 3:  # batch x F x T
            istft = torch.istft(
                stft,
                window=self.synthesis_window,
                n_fft=self.fft_length_synth,
                hop_length=self.shift_length,
                win_length=self.frame_length,
                normalized=self.normalized,
                center=self.center,
                length=length,
                return_complex=False,
            )
        elif stft.ndim == 4:  # batch x M x F x T
            istft = torch.stack(
                [
                    torch.istft(
                        x,
                        window=self.synthesis_window,
                        n_fft=self.fft_length,
                        hop_length=self.shift_length,
                        win_length=self.frame_length,
                        normalized=self.normalized,
                        center=self.center,
                        length=length,
                        return_complex=False,
                    )
                    for x in stft
                ]
            )
        else:
            raise ValueError("unsupported STFT shape!")
        return istft


class CustomBatchBase:
    def __init__(self):
        pass

    def pin_memory(self):
        self.signals = {
            key: val.pin_memory() if val is torch.Tensor else val
            for key, val in self.signals.items()
        }
        return self

    def cuda(self, device=None, non_blocking=True):
        self.signals = {
            key: val.cuda(device=device, non_blocking=non_blocking)
            for key, val in self.signals.items()
        }
        return self

    def to(self, device=None, dtype=None, non_blocking=True):
        self.signals = {
            key: val.to(device=device, dtype=dtype, non_blocking=non_blocking)
            for key, val in self.signals.items()
        }
        return self


@dataclass
class CustomBatchSignalsMeta(CustomBatchBase):
    signals: dict
    meta: list

    def __init__(self, batch: list) -> None:
        super().__init__()

        self.signals = default_collate(
            [x[0] for x in batch],
        )
        self.meta = [x[1] for x in batch]


def collate_fn_signals_meta(batch):
    return CustomBatchSignalsMeta(batch)


class CustomBatchSignalsMetaVariableLength(CustomBatchBase):
    signals: dict
    meta: list

    def __init__(self, batch: list) -> None:
        super().__init__()

        keys = batch[0][0].keys()
        self.signals = {}
        for key in keys:
            lst = [x[0][key].transpose(0, -1) for x in batch]
            self.signals[key] = torch.nn.utils.rnn.pad_sequence(
                lst, batch_first=True
            ).transpose(1, -1)
        self.meta = [x[1] for x in batch]


def collate_fn_signals_meta_variable_length(batch: list):
    return CustomBatchSignalsMetaVariableLength(batch)


def time_to_smoothing_constant(time_constant, shift_length, fs=16000):
    """convert time constant to smoothing constant"""
    return np.exp(-shift_length / (fs * time_constant))


def smoothing_to_time_constant(smoothing_constant, shift_length, fs=16000):
    """convert smoothing constant to time constant"""
    return -shift_length / (fs * np.log(smoothing_constant))


def convert_smoothing_constant(smoothing1: float, shift1: int, shift2: int) -> float:
    """convert smoothing constant to new STFT framework

    Args:
        smoothing1 (float): [original smoothing constant]
        shift1 (int): [original frame shift]
        shift2 (int): [new frame shift]

    Returns:
        float: [description]
    """
    return smoothing1 ** (shift2 / shift1)


def get_save_dir():
    path_current_file = os.path.dirname(os.path.abspath(__file__))
    str_year_month_day = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        path_current_file,
        "saved",
        f"{str_year_month_day}_" + hri.get_new_id().lower().replace(" ", "-"),
    )
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
