from copy import deepcopy
from typing_extensions import override

import pytorch_lightning as pl
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.callbacks import StochasticWeightAveraging


class StochasticWeightAveragingAndCopy(StochasticWeightAveraging):
    @override
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        trainer.fit_loop._skip_backward = False
        self.transfer_weights(self._average_model, pl_module.net_swa)

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if len(trainer.optimizers) != 1:
            raise MisconfigurationException("SWA currently works with 1 `optimizer`.")

        if len(trainer.lr_scheduler_configs) > 1:
            raise MisconfigurationException(
                "SWA currently not supported for more than 1 `lr_scheduler`."
            )

        assert trainer.max_epochs is not None
        if isinstance(self._swa_epoch_start, float):
            self._swa_epoch_start = int(trainer.max_epochs * self._swa_epoch_start)

        self._model_contains_batch_norm = self.pl_module_contains_batch_norm(pl_module)

        self._max_epochs = trainer.max_epochs
        if self._model_contains_batch_norm:
            # virtually increase max_epochs to perform batch norm update on latest epoch.
            assert trainer.fit_loop.max_epochs is not None
            trainer.fit_loop.max_epochs += 1

        if self._scheduler_state is not None:
            self._clear_schedulers(trainer)

        if not hasattr(pl_module, "net_swa"):
            pl_module.net_swa = deepcopy(pl_module.net)
