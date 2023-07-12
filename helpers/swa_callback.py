# Adapted from PyTorch Lightning so that it only does the averaging
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Stochastic Weight Averaging Callback
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
from copy import deepcopy
from typing import Callable, Optional, Union

import torch
from torch import nn

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from torch.optim.swa_utils import SWALR

_AVG_FN = Callable[[torch.Tensor, torch.Tensor, torch.LongTensor], torch.FloatTensor]


class StochasticWeightAveraging(Callback):
    def __init__(
        self,
        swa_epoch_start: Union[int, float] = 0.8,
        swa_freq: Union[int, float] = 3,
        swa_lrs: Optional[Union[float, list]] = None,
        annealing_epochs: int = 10,
        annealing_strategy: str = "cos",
        avg_fn: Optional[_AVG_FN] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        r"""

        Implements the Stochastic Weight Averaging (SWA) Callback to average a model.

        Stochastic Weight Averaging was proposed in ``Averaging Weights Leads to
        Wider Optima and Better Generalization`` by Pavel Izmailov, Dmitrii
        Podoprikhin, Timur Garipov, Dmitry Vetrov and Andrew Gordon Wilson
        (UAI 2018).

        This documentation is highly inspired by PyTorch's work on SWA.
        The callback arguments follow the scheme defined in PyTorch's ``swa_utils`` package.

        For a SWA explanation, please take a look
        `here <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging>`_.

        .. warning:: ``StochasticWeightAveraging`` is in beta and subject to change.

        .. warning:: ``StochasticWeightAveraging`` is currently not supported for multiple optimizers/schedulers.

        SWA can easily be activated directly from the Trainer as follow:

        .. code-block:: python

            Trainer(stochastic_weight_avg=True)

        Arguments:

            swa_epoch_start: If provided as int, the procedure will start from
                the ``swa_epoch_start``-th epoch. If provided as float between 0 and 1,
                the procedure will start from ``int(swa_epoch_start * max_epochs)`` epoch

            swa_lrs: the learning rate value for all param groups together or separately for each group.

            annealing_epochs: number of epochs in the annealing phase (default: 10)

            annealing_strategy: Specifies the annealing strategy (default: "cos"):

                - ``"cos"``. For cosine annealing.
                - ``"linear"`` For linear annealing

            avg_fn: the averaging function used to update the parameters;
                the function must take in the current value of the
                :class:`AveragedModel` parameter, the current value of :attr:`model`
                parameter and the number of models already averaged; if None,
                equally weighted average is used (default: ``None``)

            device: if provided, the averaged model will be stored on the ``device``.
                When None is provided, it will infer the `device` from ``pl_module``.
                (default: ``"cpu"``)

        """

        err_msg = "swa_epoch_start should be a >0 integer or a float between 0 and 1."
        if isinstance(swa_epoch_start, int) and swa_epoch_start < 1:
            raise MisconfigurationException(err_msg)
        if isinstance(swa_epoch_start, float) and not (0 <= swa_epoch_start <= 1):
            raise MisconfigurationException(err_msg)

        wrong_type = not isinstance(swa_lrs, (float, list))
        wrong_float = isinstance(swa_lrs, float) and swa_lrs <= 0
        wrong_list = isinstance(swa_lrs, list) and not all(
            lr > 0 and isinstance(lr, float) for lr in swa_lrs
        )
        if swa_lrs is not None and (wrong_type or wrong_float or wrong_list):
            raise MisconfigurationException(
                "The `swa_lrs` should be a positive float or a list of positive float."
            )

        if avg_fn is not None and not isinstance(avg_fn, Callable):
            raise MisconfigurationException("The `avg_fn` should be callable.")

        if device is not None and not isinstance(device, (torch.device, str)):
            raise MisconfigurationException(
                f"device is expected to be a torch.device or a str. Found {device}"
            )
        self.swa_freq = swa_freq
        self._swa_epoch_start = swa_epoch_start
        self._swa_lrs = swa_lrs
        self._annealing_epochs = annealing_epochs
        self._annealing_strategy = annealing_strategy
        self._avg_fn = avg_fn or self.avg_fn
        self._device = device
        self._model_contains_batch_norm = None
        self._average_model = None

    @property
    def swa_start(self) -> int:
        return max(self._swa_epoch_start - 1, 0)  # 0-based

    @property
    def swa_end(self) -> int:
        return self._max_epochs - 1  # 0-based

    @staticmethod
    def pl_module_contains_batch_norm(pl_module: "pl.LightningModule"):
        return any(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in pl_module.modules()
        )

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ):
        # copy the model before moving it to accelerator device.
        self._average_model = deepcopy(pl_module.net)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        optimizers = pl_module.optimizers()
        lr_schedulers = pl_module.lr_schedulers()

        if type(optimizers) == list:
            raise MisconfigurationException("SWA currently works with 1 `optimizer`.")

        if type(lr_schedulers) == list:
            raise MisconfigurationException(
                "SWA currently not supported for more than 1 `lr_scheduler`."
            )

        if isinstance(self._swa_epoch_start, float):
            self._swa_epoch_start = int(trainer.max_epochs * self._swa_epoch_start)

        self._model_contains_batch_norm = self.pl_module_contains_batch_norm(pl_module)

        self._max_epochs = trainer.max_epochs
        if self._model_contains_batch_norm:
            print("\n\n_model_contains_batch_norm\n\n")
            # virtually increase max_epochs to perform batch norm update on latest epoch.
            trainer.max_epochs += 1

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        if trainer.current_epoch == self.swa_start:
            print(f"\n\n SWA START at {trainer.current_epoch}\n\n")
            # move average model to request device.
            self._average_model = self._average_model.to(
                self._device or pl_module.device
            )

            optimizers = trainer.optimizers

            for param_group in optimizers[0].param_groups:
                if self._swa_lrs is None:
                    initial_lr = param_group["lr"]

                elif isinstance(self._swa_lrs, float):
                    initial_lr = self._swa_lrs

                else:
                    initial_lr = self._swa_lrs[0]

                param_group["initial_lr"] = initial_lr

            self._swa_lrs = initial_lr

            self._swa_scheduler = SWALR(
                optimizers[0],
                swa_lr=initial_lr,
                anneal_epochs=self._annealing_epochs,
                anneal_strategy=self._annealing_strategy,
                last_epoch=trainer.max_epochs
                if self._annealing_strategy == "cos"
                else -1,
            )

            self.n_averaged = torch.tensor(0, dtype=torch.long, device=pl_module.device)
            pl_module.net_swa = self._average_model
        if (self.swa_start <= trainer.current_epoch <= self.swa_end) and (
            (trainer.current_epoch - self.swa_start) % self.swa_freq == 0
        ):
            self.update_parameters(
                self._average_model, pl_module.net, self.n_averaged, self.avg_fn
            )
            pl_module.net_swa = self._average_model

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", *args
    ):
        #  skipped here. Don't know what I'm doing
        #  trainer.train_loop._skip_backward = False
        pass

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pass

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Called when the val epoch begins."""
        if (self.swa_start <= trainer.current_epoch <= self.swa_end) and (
            (trainer.current_epoch - self.swa_start) % self.swa_freq == 0
        ):
            pl_module.do_swa = True
        else:
            pl_module.do_swa = False

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        """Called when the val epoch ends."""
        pass

    @staticmethod
    def transfer_weights(
        src_pl_module: "pl.LightningModule", dst_pl_module: "pl.LightningModule"
    ):
        for src_param, dst_param in zip(
            src_pl_module.parameters(), dst_pl_module.parameters()
        ):
            dst_param.detach().copy_(src_param.to(dst_param.device))

    def reset_batch_norm_and_save_state(self, pl_module):
        """
        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L140-L154
        """
        self.momenta = {}
        for module in pl_module.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            module.running_mean = torch.zeros_like(
                module.running_mean,
                device=pl_module.device,
                dtype=module.running_mean.dtype,
            )
            module.running_var = torch.ones_like(
                module.running_var,
                device=pl_module.device,
                dtype=module.running_var.dtype,
            )
            self.momenta[module] = module.momentum
            module.momentum = None
            module.num_batches_tracked *= 0

    def reset_momenta(self):
        """
        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L164-L165
        """
        for bn_module in self.momenta.keys():
            bn_module.momentum = self.momenta[bn_module]

    @staticmethod
    def update_parameters(
        average_model, model, n_averaged: torch.LongTensor, avg_fn: _AVG_FN
    ):
        """
        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L104-L112
        """
        for p_swa, p_model in zip(average_model.parameters(), model.parameters()):
            device = p_swa.device
            p_swa_ = p_swa.detach()
            p_model_ = p_model.detach().to(device)
            src = (
                p_model_
                if n_averaged == 0
                else avg_fn(p_swa_, p_model_, n_averaged.to(device))
            )
            p_swa_.copy_(src)
        n_averaged += 1

    @staticmethod
    def avg_fn(
        averaged_model_parameter: torch.Tensor,
        model_parameter: torch.Tensor,
        num_averaged: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L95-L97
        """
        return averaged_model_parameter + (
            model_parameter - averaged_model_parameter
        ) / (num_averaged + 1)
