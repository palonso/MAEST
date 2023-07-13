import logging
import pickle
from pathlib import Path
from typing import Any, List

import numpy as np
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sklearn import metrics
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

from helpers.ramp import exp_warmup_linear_down
from config_updates import add_configs

ex = Experiment("ex_tl")

ex.observers.append(FileStorageObserver("exp_logs"))
_logger = logging.getLogger("ex_discogs")


@ex.config
def default_config():
    max_epochs = 60

    trainer = {
        "max_epochs": max_epochs,
        "devices": 1,
        "num_sanity_val_steps": 0,
    }

    optimizer = {
        "monitor": "val_roc",
        "weight_decay": 1e-3,
        "scheduler": "exp_warmup_linear_down",
        "max_lr": 1e-4,
        "max_lr_epochs": 10,
        "max_epochs": max_epochs,
        # cycliclr
        "base_lr": 1e-7,
        # exponential
        "warmup_epochs": 10,
        "gamma": 0.5,
    }

    model = {
        "drop_out": 0.5,
        "hidden_units": 512,
    }

    data = {
        "base_dir": "embeddings/mtt/30sec/no_swa/10/",
        "metadata_dir": "mtt/",
        "batch_size": 128,
        "num_workers": 16,
        "types": "c",
        "reduce": "mean",
        "token_size": 768,
        "n_classes": 50,
    }


# register extra possible configs
add_configs(ex)


class Model(pl.LightningModule):
    @ex.capture(prefix="model")
    def __init__(
        self,
        in_features,
        n_classes,
        drop_out,
        hidden_units,
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(drop_out),
            nn.Linear(hidden_units, n_classes),
        )
        self.sigmoid = nn.Sigmoid()

        self.best_checkpoint_path = "best"

        self.val_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)

        return loss

    def test_val_step(self, batch, batch_idx, outputs, key):
        x, y = batch
        y_logits = self.forward(x)
        loss = nn.functional.binary_cross_entropy_with_logits(y_logits, y)
        self.log(f"{key}_loss", loss, prog_bar=True)

        y_hat = self.sigmoid(y_logits)

        output_dict = {f"{key}_loss": loss, f"{key}_y": y, f"{key}_y_hat": y_hat}
        outputs.append(output_dict)

        return outputs

    def validation_step(self, batch, batch_idx):
        return self.test_val_step(batch, batch_idx, self.val_step_outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.test_val_step(batch, batch_idx, self.test_step_outputs, "test")

    def on_val_test_epoch_end(self, outputs: List[Any], key: str) -> None:
        y = torch.cat([output[f"{key}_y"] for output in outputs])
        y_hat = torch.cat([output[f"{key}_y_hat"] for output in outputs])

        y = y.detach().cpu().numpy().astype("int")
        y_hat = y_hat.detach().cpu().numpy()

        ap = metrics.average_precision_score(y, y_hat, average="macro")
        roc = metrics.roc_auc_score(y, y_hat, average="macro")

        self.log(f"{key}_ap", ap)
        self.log(f"{key}_roc", roc)

        outputs.clear()

    def on_validation_epoch_end(self) -> None:
        self.on_val_test_epoch_end(self.val_step_outputs, "val")

    def on_test_epoch_end(self) -> None:
        self.on_val_test_epoch_end(self.test_step_outputs, "test")

    @ex.capture(prefix="optimizer")
    def configure_optimizers(
        self,
        max_epochs,
        max_lr_epochs,
        base_lr,
        max_lr,
        scheduler,
        warmup_epochs,
        gamma,
        weight_decay,
    ):
        optimizer = optim.AdamW(self.parameters(), lr=max_lr, weight_decay=weight_decay)

        if scheduler == "cyclic":
            schedulers = [
                {
                    "scheduler": optim.lr_scheduler.CyclicLR(
                        optimizer=optimizer,
                        base_lr=base_lr,
                        max_lr=max_lr,
                        mode="triangular2",
                        step_size_up=145,
                        cycle_momentum=False,
                    ),
                    "interval": "step",
                    "frequency": 1,
                }
            ]

        elif scheduler == "exponential":
            schedulers = [
                optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=lambda e: (e + 1e-7) / warmup_epochs
                    if e < warmup_epochs
                    else 1,
                ),
                optim.lr_scheduler.ExponentialLR(
                    optimizer=optimizer,
                    gamma=gamma,
                    last_epoch=-1,
                ),
            ]

        elif scheduler == "exp_warmup_linear_down":
            schedulers = [
                optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=exp_warmup_linear_down(
                        warmup_epochs,
                        max_epochs - max_lr_epochs,
                        max_lr_epochs,
                        base_lr,
                    ),
                ),
            ]

        return [optimizer], schedulers

    @ex.capture(prefix="trainer")
    def configure_callbacks(self, monitor="val_roc"):
        if "roc" in monitor:
            mode = "max"
        elif "loss" in monitor:
            mode = "min"

        return [
            ModelCheckpoint(
                filename=self.best_checkpoint_path,
                save_top_k=1,
                monitor=monitor,
                mode=mode,
                save_weights_only=True,
                verbose=True,
            ),
            # LearningRateMonitor(logging_interval="step")
        ]


class EmbeddingDataset(Dataset):
    @ex.capture(prefix="data")
    def __init__(self, groundtruth_file, base_dir, types, reduce):
        self.base_dir = base_dir

        self.types = types
        self.reduce = reduce

        with open(groundtruth_file, "rb") as gf:
            self.groundtruth = pickle.load(gf)

        self.filenames = {
            i: filename for i, filename in enumerate(list(self.groundtruth.keys()))
        }
        self.length = len(self.groundtruth)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        filename = self.filenames[index]
        target = self.groundtruth[filename]

        embedding_path = Path(self.base_dir, filename + ".embeddings.npy")
        embedding = np.load(embedding_path)
        embedding = self.post_process(embedding)

        return embedding, target.astype("float32")

    def post_process(self, embedding):
        # todo: implement other postprocessing
        if len(embedding.shape) == 2:
            embedding = np.mean(embedding, axis=0)

        if embedding.shape[-1] == 768:
            return embedding

        embedding = embedding.reshape(3, -1)
        embedding_de = {
            "c": embedding[0],
            "d": embedding[1],
            "t": embedding[2],
        }
        embeddings = [v for k, v in embedding_de.items() if k in self.types]

        if self.reduce == "mean":
            return np.mean(np.array(embeddings), axis=0)
        elif self.reduce == "stack":
            return np.hstack(embeddings)


class DataModule(pl.LightningDataModule):
    @ex.capture(prefix="data")
    def __init__(
        self,
        base_dir,
        metadata_dir,
        batch_size,
        num_workers,
        types,
        reduce,
        token_size,
        n_classes,
        train_groundtruth_file=None,
        valid_groundtruth_file=None,
        test_groundtruth_file=None,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.metadata_dir = Path(metadata_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_file = (
            train_groundtruth_file if train_groundtruth_file else "groundtruth-train.pk"
        )
        valid_file = (
            valid_groundtruth_file
            if valid_groundtruth_file
            else "groundtruth-validation.pk"
        )
        test_file = (
            test_groundtruth_file if test_groundtruth_file else "groundtruth-test.pk"
        )

        self.train_groundtruth_file = self.metadata_dir / train_file
        self.val_groundtruth_file = self.metadata_dir / valid_file
        self.test_groundtruth_file = self.metadata_dir / test_file

        self.types = types
        self.reduce = reduce
        self.token_size = token_size
        self.n_classes = n_classes

        if self.reduce == "mean":
            self.in_features = self.token_size
        elif self.reduce == "stack":
            self.in_features = self.token_size * len(self.types)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.dataset_train = EmbeddingDataset(
                groundtruth_file=self.train_groundtruth_file,
                base_dir=self.base_dir,
            )
            self.dataset_val = EmbeddingDataset(
                groundtruth_file=self.val_groundtruth_file,
                base_dir=self.base_dir,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.dataset_test = EmbeddingDataset(
                groundtruth_file=self.test_groundtruth_file,
                base_dir=self.base_dir,
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )


@ex.automain
def tl_pipeline(_run, _config):
    _logger.info("starting transfer learning experiment")
    _logger.info(_config)

    datamodule = DataModule()
    model = Model(
        in_features=datamodule.in_features,
        n_classes=datamodule.n_classes,
    )

    tb_logger = TensorBoardLogger("exp_logs/", version=_run._id)
    trainer = pl.Trainer(logger=tb_logger, **_config["trainer"])

    trainer.fit(model=model, datamodule=datamodule)

    model.eval()
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
