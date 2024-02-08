import logging

import torch
import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint
from sacred import Ingredient
from sklearn import metrics

# from sacred.config_helpers import Ingredient, CMD
from torch.nn import functional as F

from helpers.mixup import my_mixup
from helpers.ramp import exp_warmup_linear_down, cosine_cycle
from helpers.swa_callback import StochasticWeightAveraging
from models.maest import maest

module_ing = Ingredient("module")
_logger = logging.getLogger("module")


@module_ing.config
def default_conf():
    do_swa = True
    swa_epoch_start = 50
    swa_freq = 5

    mixup_alpha = 0.3  # Set to 0 to skip

    optimizer = {
        "lr": 0.00002,  # learning rate
        "adamw": True,
        "weight_decay": 0.0001,
        "warm_up_len": 5,
        "ramp_down_start": 50,
        "ramp_down_len": 50,
        "last_lr_value": 0.01,
        "schedule_mode": "exp_lin",
        "reaload_dataloaders_every_n_epochs": 1,
    }


class Module(pl.LightningModule):
    @module_ing.capture
    def __init__(
        self,
        do_swa,
        swa_epoch_start,
        swa_freq,
        mixup_alpha,
        distributed_mode=False,
    ):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.do_swa = do_swa
        self.swa_freq = swa_freq
        self.swa_epoch_start = swa_epoch_start
        self.distributed_mode = distributed_mode

        self.net = maest()

        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, batch, transformer_block=-1):
        return self.net.forward(
            batch, transformer_block=-1, return_self_attention=False
        )

    def training_step(self, batch, batch_idx):
        x, f, y = batch
        batch_size = len(y)

        if self.mixup_alpha > 0:
            rn_indices, lam = my_mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)

            x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (
                1.0 - lam.reshape(batch_size, 1, 1, 1)
            )
            y = y * lam.reshape(batch_size, 1) + y[rn_indices] * (
                1.0 - lam.reshape(batch_size, 1)
            )

        y_hat, embed = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(y_hat, y)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, f, y = batch
        logits, embed = self.forward(x, transformer_block=self.transformer_block)

        return {
            "logits": logits.detach().cpu(),
            "embeddings": embed.detach().cpu(),
            "filename": f,
        }

    def set_prediction_tranformer_block(self, transformer_block):
        self.transformer_block = transformer_block

    @staticmethod
    def _join(strings):
        return "_".join(filter(lambda x: x, strings))

    def test_validation_step(self, batch, batch_idx, output_buffer, stage):
        x, f, y = batch
        outputs = {"y": y.detach()}
        batch_size = len(y)

        net_map = [(None, self.net)]
        if self.do_swa:
            net_map.append(("swa", self.net_swa))

        for name, net in net_map:
            logits, _ = net(x)
            loss = F.binary_cross_entropy_with_logits(logits, y).mean()
            y_hat = torch.sigmoid(logits.detach())

            outputs[self._join((name, "loss"))] = loss
            outputs[self._join((name, "y_hat"))] = y_hat
            output_buffer.append(outputs)

            self.log(
                self._join((stage, "loss", name)),
                loss,
                batch_size=batch_size,
                sync_dist=True,
            )

        return outputs

    def validation_step(self, batch, batch_idx):
        return self.test_validation_step(
            batch, batch_idx, self.validation_outputs, "val"
        )

    def test_step(self, batch, batch_idx):
        return self.test_validation_step(batch, batch_idx, self.test_outputs, "test")

    def on_test_validation_epoch_end(self, outputs, stage):
        net_map = [(None, self.net)]
        if self.do_swa:
            net_map.append(("swa", self.net_swa))

        y = torch.cat([x["y"] for x in outputs], dim=0)

        if self.distributed_mode:
            y = self.all_gather(y).reshape(-1, y.shape[-1])

        y = y.cpu().numpy()
        _logger.debug("end of validation")
        _logger.debug(f"outputs len: {len(outputs)}")

        _logger.debug(f"y shape: {y.shape}")

        for name, net in net_map:
            loss = torch.stack([x[self._join((name, "loss"))] for x in outputs]).mean()
            y_hat = torch.cat([x[self._join((name, "y_hat"))] for x in outputs], dim=0)

            _logger.debug(f"y_hat shape: {y_hat.shape}")

            if self.distributed_mode:
                loss = self.all_gather(loss)
                y_hat = self.all_gather(y_hat).reshape(-1, y_hat.shape[-1])

            # detach tensors
            loss = loss.cpu().numpy().mean()
            y_hat = y_hat.cpu().numpy()

            samples_per_class = np.sum(y, axis=0)
            _logger.debug(samples_per_class)
            _logger.debug(f"n val/test samples: {sum(samples_per_class)}")

            ap = metrics.average_precision_score(y, y_hat, average="macro")
            roc = metrics.roc_auc_score(y, y_hat, average="macro")

            self.log_dict(
                {
                    self._join((stage, "loss", name)): loss,
                    self._join((stage, "ap", name)): ap,
                    self._join((stage, "roc", name)): roc,
                },
                sync_dist=True,
            )

        outputs.clear()

    def on_validation_epoch_end(self):
        self.on_test_validation_epoch_end(self.validation_outputs, "val")

        if self.trainer.num_nodes > 1:
            self.trainer.strategy.barrier()

    def on_test_epoch_end(self):
        self.on_test_validation_epoch_end(self.test_outputs, "test")

    @staticmethod
    @module_ing.capture(prefix="optimizer")
    def get_scheduler_lambda(
        warm_up_len, ramp_down_start, ramp_down_len, last_lr_value, schedule_mode
    ):
        if schedule_mode == "exp_lin":
            return exp_warmup_linear_down(
                warm_up_len, ramp_down_len, ramp_down_start, last_lr_value
            )
        if schedule_mode == "cos_cyc":
            return cosine_cycle(warm_up_len, ramp_down_start, last_lr_value)
        raise RuntimeError(
            f"schedule_mode={schedule_mode} Unknown for a lambda funtion."
        )

    @staticmethod
    @module_ing.capture(prefix="optimizer")
    def get_lr_scheduler(optimizer, schedule_mode):
        if schedule_mode in {"exp_lin", "cos_cyc"}:
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, Module.get_scheduler_lambda()
            )
        raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")

    @staticmethod
    @module_ing.capture(prefix="optimizer")
    def get_optimizer(params, lr, adamw, weight_decay):
        if adamw:
            _logger.debug(f"Using adamw weight_decay={weight_decay}!")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.Adam(params, lr=lr)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = self.get_optimizer(self.parameters())
        # torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.get_lr_scheduler(optimizer),
        }

    def configure_callbacks(self):
        callbacks = []
        monitor = "val_loss"
        callbacks.append(
            ModelCheckpoint(
                monitor=monitor, mode="min", filename="{epoch}-{val_loss:.2f}-best"
            )
        )
        _logger.debug(f"Adding checkpoint monitoring {monitor}")

        if self.do_swa:
            callbacks.append(
                StochasticWeightAveraging(
                    swa_epoch_start=self.swa_epoch_start, swa_freq=self.swa_freq
                )
            )
            _logger.debug("Using swa!")

        return callbacks


class TeacherStudentModule(Module):
    def training_step(self, batch, batch_idx):
        x, f, y, y_teacher = batch
        batch_size = len(y)

        if self.mixup_alpha > 0:
            rn_indices, lam = my_mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (
                1.0 - lam.reshape(batch_size, 1, 1, 1)
            )
            y = y * lam.reshape(batch_size, 1) + y[rn_indices] * (
                1.0 - lam.reshape(batch_size, 1)
            )
            y_teacher = y_teacher * lam.reshape(batch_size, 1) + y_teacher[
                rn_indices
            ] * (1.0 - lam.reshape(batch_size, 1))

        y_hat, y_hat_teacher, _ = self.forward(x)

        loss_standard = F.binary_cross_entropy_with_logits(y_hat, y)
        loss_teacher = F.binary_cross_entropy_with_logits(y_hat_teacher, y_teacher)
        loss = (loss_standard + loss_teacher) / 2

        results = {
            "train_loss": loss,
            "train_loss_standard": loss_standard,
            "tran_loss_teacher": loss_teacher,
        }

        self.log_dict(
            results,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_validation_step(self, batch, batch_idx, output_buffer, stage):
        x, f, y, y_teacher = batch
        outputs = {"y": y.detach(), "y_teacher": y_teacher.detach()}

        net_map = [(None, self.net)]
        if self.do_swa:
            net_map.append(("swa", self.net_swa))

        for name, net in net_map:
            logits, _ = net(x)
            loss_standard = F.binary_cross_entropy_with_logits(logits, y).mean()
            loss_teacher = F.binary_cross_entropy_with_logits(logits, y_teacher).mean()
            loss = (loss_standard + loss_teacher) / 2

            y_hat = torch.sigmoid(logits.detach())

            outputs[self._join((name, "loss_standard"))] = loss_standard
            outputs[self._join((name, "loss_teacher"))] = loss_teacher
            outputs[self._join((name, "loss"))] = loss
            outputs[self._join((name, "y_hat"))] = y_hat

            output_buffer.append(outputs)

            self.log_dict(
                {
                    self._join((stage, "loss_standard", name)): loss_standard,
                    self._join((stage, "loss_teacher", name)): loss_teacher,
                    self._join((stage, "loss", name)): loss,
                }
            )

        return outputs
