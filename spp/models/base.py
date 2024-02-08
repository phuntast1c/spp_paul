from abc import abstractmethod
from typing import Any, Dict, Sequence, Union

import lightning as pl
import matplotlib.pyplot as plt
import torch
import wandb
from lightning.pytorch.callbacks.callback import Callback
from torchmetrics import MetricCollection

from .. import metrics as metrics
from .. import utils


class BaseLitModel(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        batch_size: int,
        metrics_test: Union[tuple, str],
        metrics_val: Union[tuple, str],
        model_name: str,
        my_optimizer: str = "AdamW",
        my_lr_scheduler: str = "ReduceLROnPlateau",
        **kwargs,
    ):
        super().__init__()

        self.learning_rate = lr
        self.batch_size = batch_size
        self.model_name = model_name
        self.my_optimizer = my_optimizer
        self.my_lr_scheduler = my_lr_scheduler

        self.nan_batch_counter = 0.0

        if isinstance(metrics_test, str):
            metrics_test = metrics_test.split(",")
        if isinstance(metrics_val, str):
            metrics_val = metrics_val.split(",")
        self.metric_collections = {
            "test": MetricCollection(
                [getattr(metrics, met)() for met in metrics_test if met != ""]
            ),
            "val": MetricCollection(
                [getattr(metrics, met)() for met in metrics_val if met != ""]
            ),
        }

        self.test_outputs = []

    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self):
        if self.my_optimizer == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.my_optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError

        if self.my_lr_scheduler == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, factor=0.5
            )
        elif self.my_lr_scheduler == "OneCycleLR":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    total_steps=self.trainer.estimated_stepping_batches,
                    verbose=True,
                ),
                "interval": "step",
            }
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val/loss",
        }

    # skip batches including NaN gradients
    def on_after_backward(self) -> None:
        increase_nan_batch_counter = False
        for param in self.parameters():
            if param.grad is not None:
                nan_grads = torch.isnan(param.grad)
                if torch.any(nan_grads):
                    param.grad[nan_grads] = 0.0
                    increase_nan_batch_counter = True
        if increase_nan_batch_counter:
            self.nan_batch_counter += 1

        self.log(
            "ptl/nan_batch_counter",
            self.nan_batch_counter,
        )
        return super().on_after_backward()

    def training_step(self, batch, idx):
        self.log(
            "learning rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=True,
            on_epoch=False,
        )
        output = self(batch.signals)
        loss = self.loss(output, batch.signals, batch.meta)
        self.log_dict(
            {f"train/{x}": y for x, y in loss.items() if "loss" in x},
        )
        return {"loss": loss["loss"]}

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # delete early stopping callback state
        checkpoint["callbacks"] = {
            x: y for x, y in checkpoint["callbacks"].items() if "EarlyStopping" not in x
        }
        # delete optimizer and LR scheduler states
        checkpoint["optimizer_states"][0]["param_groups"][0]["lr"] = self.learning_rate
        checkpoint["lr_schedulers"][0]["best"] = 1e6
        return super().on_load_checkpoint(checkpoint)

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        return super().configure_callbacks()

    def validation_step(self, batch, idx):
        output = self(batch.signals)
        loss = self.loss(output, batch.signals, batch.meta)
        self.log_dict(
            {f"val/{x}": y for x, y in loss.items() if "loss" in x},
        )
        metrics_dict = {}
        for metric in self.metric_collections["val"]:
            (
                metrics_dict["metrics/val/enh_" + metric.__name__.upper()],
                metrics_dict["metrics/val/" + metric.__name__.upper()],
            ) = utils.get_measure_enhanced_noisy(output, batch.signals, metric)

        self.log_dict(metrics_dict)
        if idx == 0:
            self.plot_spp(batch, loss)

        return {"loss_val": loss["loss"], "metrics": metrics_dict}

    def plot_spp(self, batch, loss):
        spp_oracle = loss["spp_oracle"][0].detach().cpu().numpy()
        spp_estimate = loss["spp_estimate"][0].detach().cpu().numpy()

        fig, ax = plt.subplots()
        im = ax.matshow(
            spp_oracle, cmap="magma", aspect="auto", origin="lower", vmin=0, vmax=1
        )
        ax.set_title("SPP Oracle")
        fig.colorbar(im)
        wandb.log({"images/spp_oracle": fig, "trainer/global_step": self.global_step})

        fig, ax = plt.subplots()
        im = ax.matshow(
            spp_estimate,
            cmap="magma",
            aspect="auto",
            origin="lower",
            vmin=0,
            vmax=1,
        )
        ax.set_title("SPP Estimate")
        fig.colorbar(im)
        wandb.log({"images/spp_estimate": fig, "trainer/global_step": self.global_step})

        fig, ax = plt.subplots()
        speech_periodogram = (
            10
            * (self.stft.get_stft(batch.signals["clean"])[0, 0].abs().pow(2).log10())
            .detach()
            .cpu()
            .numpy()
        )
        im = ax.matshow(speech_periodogram, cmap="magma", aspect="auto", origin="lower")
        ax.set_title("Speech Periodogram")
        fig.colorbar(im)
        wandb.log({"images/speech_psd": fig, "trainer/global_step": self.global_step})

        plt.close("all")

    def train_dataloader(self):
        return self.datamodule.train_dataloader()
