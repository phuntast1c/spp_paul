import os

import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from spp import utils
from spp.models import base

torch.set_float32_matmul_precision("high")

SAVE_DIR = utils.get_save_dir()


class MyLightningCLI(LightningCLI):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def before_instantiate_classes(self) -> None:
        wandb.init(
            project=self.config[self.subcommand].trainer.logger.init_args.project,
            dir=SAVE_DIR,
            notes=SAVE_DIR,
            job_type=self.subcommand,
            name=SAVE_DIR.split("_")[-1],
        )
        return super().before_instantiate_classes()

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="specify wandb run path to continue training from a previous run",
        )
        parser.add_argument(
            "--job_type",
            type=str,
            default="fit",
        )

        parser.link_arguments("model.init_args.batch_size", "data.init_args.batch_size")
        parser.link_arguments("data.init_args.fs", "model.init_args.fs")
        parser.link_arguments("data.init_args.fs", "model.init_args.fs")


def main():
    print(f"saving to {SAVE_DIR}")
    _ = MyLightningCLI(
        model_class=base.BaseLitModel,
        subclass_mode_model=True,
        seed_everything_default=1337,
        save_config_kwargs={"config_filename": os.path.join(SAVE_DIR, "config.yaml")},
        run=True,
        auto_configure_optimizers=False,
        trainer_defaults={
            "num_sanity_val_steps": 1,
            "log_every_n_steps": 50,
            "enable_progress_bar": True,
            "gradient_clip_val": 5.0,
            "deterministic": False,
            "benchmark": True,
            "devices": 1,
            "accelerator": "gpu",
            "callbacks": [
                ModelCheckpoint(
                    monitor="val/loss",
                    save_last=True,
                    dirpath=SAVE_DIR,
                    verbose=True,
                ),
                EarlyStopping(patience=10, monitor="val/loss"),
            ],
            "default_root_dir": SAVE_DIR,
            "strategy": "ddp_find_unused_parameters_true",
        },
    )


if __name__ == "__main__":
    main()
