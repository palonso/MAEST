import os
import logging
from collections import defaultdict
from itertools import chain
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from sacred import Experiment
from sacred.observers import FileStorageObserver

from torch.nn import functional as F
from tqdm import tqdm

from config_updates import add_configs
from models.maest import maest_ing
from models.module import Module, module_ing
from discogs.dataset import dataset_ing
from discogs.datamodule import (
    DiscogsDataModule,
    datamodule_ing,
)

ex = Experiment(
    "ex_maest",
    ingredients=[
        dataset_ing,
        datamodule_ing,
        maest_ing,
        module_ing,
    ],
)
ex.observers.append(FileStorageObserver("exp_logs"))
_logger = logging.getLogger("ex_maest")


@ex.config
def default_conf():
    process_id = os.getpid()
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

    trainer = {
        "max_epochs": 130,
        "devices": 1,
        "sync_batchnorm": True,
        "precision": "16-mixed",
        "limit_train_batches": None,
        "limit_val_batches": None,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 50,
        "reload_dataloaders_every_n_epochs": 1,
        "strategy": "ddp_find_unused_parameters_true",
        "default_root_dir": "exp_logs",
    }

    predict = {
        "transformer_block": 11,
        "out_dir": "predictions",
    }


# register extra possible configs
add_configs(ex)


@ex.command
def main(_run, _config, _log, _rnd, _seed):
    _logger.info(_config)

    tb_logger = TensorBoardLogger("exp_logs/", version=_run._id)
    trainer = pl.Trainer(logger=tb_logger, **_config["trainer"])

    distributed_mode = False
    if _config["trainer"]["devices"] > 1:
        distributed_mode = True

    module = Module(distributed_mode=distributed_mode)
    data = DiscogsDataModule()

    trainer.fit(module, data)
    return {"done": True}


@ex.command
def test(_run, _config, _log, _rnd, _seed):
    trainer = pl.Trainer(**_config["trainer"])

    module = Module()
    module.do_swa = False

    data = DiscogsDataModule()

    trainer.test(module, data)
    return {"done": True}


@ex.command
def model_speed_test(_run, _config, _log, _rnd, _seed, speed_test_batch_size=100):
    modul = Module()
    modul = modul.cuda()
    batch_size = speed_test_batch_size
    print(f"\nBATCH SIZE : {batch_size}\n")
    test_length = 100
    print(f"\ntest_length : {test_length}\n")

    x = torch.ones([batch_size, 1, 128, 998]).cuda()
    target = torch.ones([batch_size, 400]).cuda()
    # one passe
    net = modul.net
    # net(x)
    scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True
    # net = torch.jit.trace(net,(x,))
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    print("warmup")
    import time

    torch.cuda.synchronize()
    t1 = time.time()
    for i in range(10):
        with torch.cuda.amp.autocast():
            y_hat, embed = net(x)
            loss = F.binary_cross_entropy_with_logits(
                y_hat, target, reduction="none"
            ).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    t2 = time.time()
    print("warmup done:", (t2 - t1))
    torch.cuda.synchronize()
    t1 = time.time()
    print("testing speed")

    for i in range(test_length):
        with torch.cuda.amp.autocast():
            y_hat, embed = net(x)
            loss = F.binary_cross_entropy_with_logits(
                y_hat, target, reduction="none"
            ).mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    t2 = time.time()
    print("test done:", (t2 - t1))
    print("average speed: ", (test_length * batch_size) / (t2 - t1), " specs/second")


def predict(_run, _config, _log, _rnd, _seed, output_name=""):
    _logger.debug(f"extracting output: {output_name}")

    trainer = pl.Trainer(**_config["trainer"])
    module = Module()
    module.set_prediction_tranformer_block(_config["predict"]["transformer_block"])
    module.eval()

    data = DiscogsDataModule()

    outputs = trainer.predict(module, data)

    filenames = list(chain.from_iterable([x["filename"] for x in outputs]))
    _logger.debug(f"n filenames: {len(filenames)}")

    out = np.vstack([x[output_name] for x in outputs])
    _logger.debug(f"n {output_name}: {len(out)}")

    agg_out = defaultdict(list)
    for o, f in zip(out, filenames):
        agg_out[f].append(o)
    agg_out = {f: np.array(o) for f, o in agg_out.items()}

    # get ouptut directory
    subdir1 = str(_config["datamodule"]["clip_length"]) + "sec"

    subdir2 = ""
    for po_dim in ("f", "t"):
        for po_type in ("indices", "interleaved"):
            if _config["maest"][f"s_patchout_{po_dim}_{po_type}"]:
                removed_bands = "_".join(
                    np.array(_config["maest"][f"s_patchout_{po_dim}_{po_type}"]).astype(
                        "str"
                    )
                )
                subdir2 += f"_patchout_{po_dim}_{po_type}" + removed_bands

    subdir3 = str(_config["predict"]["transformer_block"])

    out_dir = Path(_config["predict"]["out_dir"]) / subdir1 / subdir2 / subdir3

    # write output files
    for k, v in agg_out.items():
        file_path = out_dir / (k + f".{output_name}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, v)


@ex.command
def extract_embeddings(_run, _config, _log, _rnd, _seed):
    predict(_run, _config, _log, _rnd, _seed, output_name="embeddings")


@ex.command
def extract_logits(_run, _config, _log, _rnd, _seed):
    predict(_run, _config, _log, _rnd, _seed, output_name="logits")


@ex.command
def compute_norm_stats(_run, _config, _log, _rnd, _seed):
    # force overriding the config, not logged = not recommended
    loader = ex.get_train_dataloaders()
    mean = []
    std = []

    for i, (audio_input, _, _) in tqdm(enumerate(loader), total=len(loader)):
        audio_input = audio_input.type(torch.DoubleTensor)
        cur_mean = torch.mean(audio_input)
        cur_std = torch.std(audio_input)
        mean.append(cur_mean)
        std.append(cur_std)
    _log.info(np.mean(mean), np.mean(std))


@ex.automain
def default_command():
    return main()
