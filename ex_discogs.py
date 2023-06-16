import os
from collections import defaultdict
from itertools import chain
from pathlib import Path
from datetime import datetime
import sys

import torch
import numpy as np
import lightning.pytorch as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger, CSVLogger
from sacred import Experiment, Ingredient

# from sacred.config_helpers import Ingredient, CMD
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_updates import add_configs
from helpers.mixup import my_mixup
from helpers.models_size import count_non_zero_params
from helpers.ramp import exp_warmup_linear_down, cosine_cycle
from helpers.workersinit import worker_init_fn
from helpers.spec_masking import SpecMasking
from sklearn import metrics
from models.passt import get_model
from discogs.dataset import (
    get_train_set,
    get_ft_weighted_sampler,
    get_val_set,
    get_test_set,
    get_predict_set,
    discogs_dataset
)

ex = Experiment("discogs", ingredients=[discogs_dataset])


@ex.config
def default_conf():
    process_id = os.getpid()
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")

    net = {
        "arch": "passt_deit_bd_p16_384",
        "n_classes": 400,
        "s_patchout_t": 30,
        "s_patchout_f": 3,
        "fstride": 10,
        "tstride": 10,
        "input_fdim": 96,
        "input_tdim": 625,
        "use_swa": True,
    }

    mel = {
        "instance_cmd": "AugmentMelSTFT",
        "n_mels": 96,
        "sr": 16000,
        "win_length": 512,
        "hopsize": 256,
        "n_fft": 512,
        "fmin": 0.0,
        "fmax": None,
        "norm": 1,
        "fmin_aug_range": 10,
        "fmax_aug_range": 2000,
    }

    dataset = {
        "clip_length": 10,
    }

    trainer = {
        "max_epochs": 130,
        "devices": 1,
        "num_sanity_val_steps": 200,
        "sync_batchnorm": True,
        "precision": 16,
        #  "benchmark": True,
        # "reload_dataloaders_every_epoch": True,
    }

    model = {
        "use_mixup": True,
        "use_masking": True,
        "mixup_alpha": 0.3,
        "do_swa": True,
        "weights_summary": "full",
        "benchmark": True,
        "reload_dataloaders_every_epoch": True,
        "sync_batchnorm": True,
    }

    optimizer = {
        "lr": 0.00002,  # learning rate
        "adamw": True,
        "weight_decay": 0.0001,
        "warm_up_len": 5,
        "ramp_down_start": 50,
        "ramp_down_len": 50,
        "last_lr_value": 0.01,
        "schedule_mode": "exp_lin",
    }

    dataloader_train = {
        "batch_size": 12,
        "num_workers": 16,
    }

    dataloader_test = {
        "batch_size": 20,
        "num_workers": 16,
    }

# register extra possible configs
add_configs(ex)




class Module(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Module configuration
        self.use_mixup = self.config["model"]["use_mixup"]
        self.mixup_alpha = self.config["model"]["mixup_alpha"]
        self.use_masking = self.config["model"]["use_masking"]
        self.do_swa = self.config["model"]["do_swa"]
        self.distributed_mode = self.config["trainer"]["devices"] > 1
        self.teacher_target = None

        self.net = get_model()

        if self.use_masking:
            self.masking = SpecMasking()

        # desc, sum_params, sum_non_zero = count_non_zero_params(self.net)
        # self.experiment.info["start_sum_params"] = sum_params
        # self.experiment.info["start_sum_params_non_zero"] = sum_non_zero

    def training_step(self, batch, batch_idx):
        x, f, y = batch
        batch_size = len(y)

        rn_indices, lam = None, None
        if self.use_mixup:
            rn_indices, lam = my_mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (
                1.0 - lam.reshape(batch_size, 1, 1, 1)
            )

        if self.use_masking:
            x = self.masking.compute(x)

        y_hat, embed = self.forward(x)

        if self.use_mixup:
            y_mix = y * lam.reshape(batch_size, 1) + y[rn_indices] * (
                1.0 - lam.reshape(batch_size, 1)
            )
            samples_loss = F.binary_cross_entropy_with_logits(
                y_hat, y_mix, reduction="none"
            )
            loss = samples_loss.mean()
        else:
            samples_loss = F.binary_cross_entropy_with_logits(
                y_hat, y, reduction="none"
            )
            loss = samples_loss.mean()

        results = {
            "loss": loss,
        }
        self.log_dict(results, sync_dist=True)

        return results

    def on_train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        logs = {"train.loss": avg_loss, "step": self.current_epoch}

        self.log_dict(logs, sync_dist=True)

    def predict(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, f, y = batch

        if self.config.inference.n_block < 11:
            embeddings = self.net.forward_until_block(
                x,
                n_block=self.config.inference.n_block,
                return_self_attention=False,
                compact_features=True,
            )
            results = {"embeddings": embeddings.detach()}

        else:
            logits, embed = self.net(x)
            results = {
                "logits": logits.detach(),
                "embeddings": embed.detach(),
            }

            self.log_dict(results, sync_dist=True)

        results = {k: v.cpu() for k, v in results.items()}
        results["filename"] = f
        return results

    def validation_step(self, batch, batch_idx, stage="val"):
        if len(batch) == 3:
            x, f, y = batch
        elif len(batch) == 4:
            x, f, y, y_teacher = batch

        results = {}
        if self.teacher_target:
            model_name = [("", self.net)]
            if self.do_swa:
                model_name = model_name + [("swa_", self.net_swa)]
            for net_name, net in model_name:
                y_hat, y_teacher_hat, _ = net(x)
                samples_standard_loss = F.binary_cross_entropy_with_logits(y_hat, y)
                samples_teacher_loss = F.binary_cross_entropy_with_logits(
                    y_teacher_hat, y_teacher
                )
                # for validation report loss and metrics on the original subset for comparision.
                loss = samples_standard_loss.mean()
                # loss_standard = samples_standard_loss.mean()
                loss_teacher = samples_teacher_loss.mean()

                # samples_loss = (samples_standard_loss + samples_teacher_loss) / 2
                out = torch.sigmoid(y_hat.detach())
                # out_teacher = torch.sigmoid(y_teacher_hat.detach())
                # late fusion as discussed in the paper
                # out = (out_standard + out_teacher) / 2
                results = {
                    **results,
                    net_name + f"{stage}_loss": loss,
                    # net_name + f"{stage}_loss_standard": loss_standard,
                    net_name + f"{stage}_loss_teacher": loss_teacher,
                    net_name + "out": out,
                    net_name + "target": y.detach(),
                }

        else:
            results = {}
            model_name = [("", self.net)]
            if self.do_swa:
                model_name = model_name + [("swa_", self.net_swa)]
            for net_name, net in model_name:
                y_hat, _ = net(x)
                samples_loss = F.binary_cross_entropy_with_logits(y_hat, y)
                loss = samples_loss.mean()
                out = torch.sigmoid(y_hat.detach())
                results = {
                    **results,
                    net_name + f"{stage}_loss": loss,
                    net_name + "out": out,
                    net_name + "target": y.detach(),
                }

        self.log_dict(results, sync_dist=True)
        results = {k: v.cpu() for k, v in results.items()}
        return results

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, stage="test")

    def on_validation_epoch_end(self, outputs, stage="val"):
        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]
        for net_name, net in model_name:
            avg_loss = torch.stack(
                [x[net_name + f"{stage}_loss"] for x in outputs]
            ).mean()
            out = torch.cat([x[net_name + "out"] for x in outputs], dim=0)
            target = torch.cat([x[net_name + "target"] for x in outputs], dim=0)

            if not self.distributed_mode:
                try:
                    average_precision = metrics.average_precision_score(
                        target.float().numpy(), out.float().numpy(), average=None
                    )
                except ValueError:
                    average_precision = np.array([np.nan] * self.net.n_classes)
                try:
                    roc = metrics.roc_auc_score(
                        target.numpy(), out.numpy(), average=None
                    )
                except ValueError:
                    roc = np.array([np.nan] * self.net.n_classes)
                logs = {
                    net_name + f"{stage}_loss": torch.as_tensor(avg_loss).cuda(),
                    net_name
                    + f"{stage}_ap": torch.as_tensor(average_precision.mean()).cuda(),
                    net_name + f"{stage}_roc": torch.as_tensor(roc.mean()).cuda(),
                    "step": torch.as_tensor(self.current_epoch).cuda(),
                }
                self.log_dict(logs)

            if self.distributed_mode:
                allout = self.all_gather(out)
                alltarget = self.all_gather(target)
                alltarget = alltarget.reshape(-1, alltarget.shape[-1]).cpu().numpy()
                allout = allout.reshape(-1, allout.shape[-1]).cpu().numpy()

                average_precision = metrics.average_precision_score(
                    alltarget, allout, average=None
                )
                roc = metrics.roc_auc_score(alltarget, allout, average=None)
                if self.trainer.is_global_zero:
                    logs = {
                        net_name
                        + f"{stage}_ap": torch.as_tensor(
                            average_precision.mean()
                        ).cuda(),
                        net_name + f"{stage}_roc": torch.as_tensor(roc.mean()).cuda(),
                        "step": torch.as_tensor(self.current_epoch).cuda(),
                    }
                    self.log_dict(logs, sync_dist=False)

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs, stage="test")

    @staticmethod
    @ex.capture(prefix="optimizer")
    def get_scheduler_lambda(warm_up_len, ramp_down_start, ramp_down_len, last_lr_value, schedule_mode):
        if schedule_mode == "exp_lin":
            return exp_warmup_linear_down(
                warm_up_len, ramp_down_len, ramp_down_start, last_lr_value
            )
        if schedule_mode == "cos_cyc":
            return cosine_cycle(warm_up_len, ramp_down_start, last_lr_value)
        raise RuntimeError(f"schedule_mode={schedule_mode} Unknown for a lambda funtion.")

    @staticmethod
    @ex.capture(prefix="optimizer")
    def get_lr_scheduler(optimizer, schedule_mode):
        if schedule_mode in {"exp_lin", "cos_cyc"}:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, Module.get_scheduler_lambda())
        raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")

    @staticmethod
    @ex.capture(prefix="optimizer")
    def get_optimizer(params, lr, adamw, weight_decay):
        if adamw:
            print(f"\nUsing adamw weight_decay={weight_decay}!\n")
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.Adam(params, lr=lr)

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        optimizer = self.get_optimizer(self.parameters())
        # torch.optim.Adam(self.parameters(), lr=self.config.lr)
        return {"optimizer": optimizer, "lr_scheduler": self.get_lr_scheduler(optimizer)}

    def configure_callbacks(self):
        return get_extra_checkpoint_callback() + get_extra_swa_callback()

    @ex.capture(prefix="dataloader_train")
    def train_dataloader(self, batch_size, num_workers):
        return DataLoader(
            dataset=get_train_set(),
            sampler=get_ft_weighted_sampler(),
            train=True,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=None,
        )

    @ex.capture(prefix="dataloader_test")
    def val_dataloader(self, batch_size, num_workers):
        return DataLoader(
            dataset=get_val_set(),
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @ex.capture(prefix="dataloader_test")
    def test_dataloader(self, batch_size, num_workers):
        return DataLoader(
            dataset=get_test_set(),
            batch_size=batch_size,
            num_workers=num_workers,
        )

    @ex.capture(prefix="dataloader_test")
    def predict_dataloader(self, batch_size, num_workers):
        return DataLoader(
            dataset=get_predict_set(),
            batch_size=batch_size,
            num_workers=num_workers,
        )


class TeacherStudentModule(Module):
    def training_step(self, batch, batch_idx):
        x, f, y, y_teacher = batch

        batch_size = len(y)

        rn_indices, lam = None, None
        if self.use_mixup:
            rn_indices, lam = my_mixup(batch_size, self.mixup_alpha)
            lam = lam.to(x.device)
            x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (
                1.0 - lam.reshape(batch_size, 1, 1, 1)
            )

        if self.use_masking:
            x = self.masking.compute(x)

        y_hat, y_hat_teacher, _ = self.forward(x)

        if self.use_mixup:
            y = y * lam.reshape(batch_size, 1) + y[rn_indices] * (
                1.0 - lam.reshape(batch_size, 1)
            )
            y_teacher = y_teacher * lam.reshape(batch_size, 1) + y_teacher[
                rn_indices
            ] * (1.0 - lam.reshape(batch_size, 1))

        samples_standard_loss = F.binary_cross_entropy_with_logits(
            y_hat, y, reduction="none"
        )
        samples_teacher_loss = F.binary_cross_entropy_with_logits(
            y_hat_teacher, y_teacher, reduction="none"
        )

        loss_standard = samples_standard_loss.mean()
        loss_teacher = samples_teacher_loss.mean()

        samples_loss = (samples_standard_loss + samples_teacher_loss) / 2
        loss = samples_loss.mean()

        results = {
            "loss": loss,
            "loss_standard": loss_standard,
            "loss_teacher": loss_teacher,
        }

        return results

    def validation_step(self, batch, batch_idx, stage="val"):
        x, f, y, y_teacher = batch

        results = {}
        model_name = [("", self.net)]
        if self.do_swa:
            model_name = model_name + [("swa_", self.net_swa)]
        for net_name, net in model_name:
            y_hat, y_teacher_hat, _ = net(x)
            samples_standard_loss = F.binary_cross_entropy_with_logits(y_hat, y)
            samples_teacher_loss = F.binary_cross_entropy_with_logits(
                y_teacher_hat, y_teacher
            )
            # for validation report loss and metrics on the original subset for comparision.
            loss = samples_standard_loss.mean()
            # loss_standard = samples_standard_loss.mean()
            loss_teacher = samples_teacher_loss.mean()

            # samples_loss = (samples_standard_loss + samples_teacher_loss) / 2
            out = torch.sigmoid(y_hat.detach())
            # out_teacher = torch.sigmoid(y_teacher_hat.detach())
            # late fusion as discussed in the paper
            # out = (out_standard + out_teacher) / 2
            results = {
                **results,
                net_name + f"{stage}_loss": loss,
                # net_name + f"{stage}_loss_standard": loss_standard,
                net_name + f"{stage}_loss_teacher": loss_teacher,
                net_name + "out": out,
                net_name + "target": y.detach(),
            }

        self.log_dict(results, sync_dist=True)
        results = {k: v.cpu() for k, v in results.items()}
        return results


def get_extra_checkpoint_callback(save_last_n=None):
    if save_last_n is None:
        return []
    return [
        ModelCheckpoint(
            monitor="step", verbose=True, save_top_k=save_last_n, mode="max"
        )
    ]


def get_extra_swa_callback(swa=True, swa_epoch_start=50, swa_freq=5):
    if not swa:
        return []
    print("\n Using swa!\n")
    from helpers.swa_callback import StochasticWeightAveraging

    return [
        StochasticWeightAveraging(swa_epoch_start=swa_epoch_start, swa_freq=swa_freq)
    ]


@ex.command
def main(_run, _config, _log, _rnd, _seed):
    module = Module(_config)
    trainer = pl.Trainer(**_config["trainer"])
    trainer.fit(module)

    return {"done": True}


@ex.command
def model_speed_test(_run, _config, _log, _rnd, _seed, speed_test_batch_size=100):

    modul = M(ex)
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


@ex.command
def evaluate_only(_run, _config, _log, _rnd, _seed):
    # force overriding the config, not logged = not recommended
    trainer = ex.get_trainer()
    train_loader = ex.get_train_dataloaders()
    val_loader = ex.get_val_dataloaders()
    modul = M(ex)
    modul.val_dataloader = None
    trainer.val_dataloaders = None
    print(f"\n\nValidation len={len(val_loader)}\n")
    res = trainer.validate(modul, val_dataloaders=val_loader)
    print("\n\n Validtaion:")
    print(res)


@ex.command
def extract_output(_run, _config, _log, _rnd, _seed, output_name=""):
    trainer = ex.get_trainer()
    modul = M(ex)
    modul.eval()

    outputs = trainer.predict(modul)

    filenames = list(chain.from_iterable([x["filename"] for x in outputs]))
    print("n filenames:", len(filenames))
    for output in [output_name]:
        print("processing output", output)
        out = np.vstack([x[output] for x in outputs])

        print(f"n {output}:", len(out))

        agg_out = defaultdict(list)
        for o, f in zip(out, filenames):
            agg_out[f].append(o)

        agg_out = {k: np.array(o) for k, o in agg_out.items()}
        subdir1 = str(_config["basedataset"]["clip_length"]) + "sec"
        subdir2 = "swa" if _config["models"]["net"]["use_swa"] else "no_swa"
        if _config["models"]["net"]["s_patchout_f_indices"]:
            removed_bands = "_".join(
                np.array(_config["models"]["net"]["s_patchout_f_indices"]).astype("str")
            )
            subdir2 += f"_patchout_f_indices" + removed_bands
        if _config["models"]["net"]["s_patchout_t_indices"]:
            removed_bands = "_".join(
                np.array(_config["models"]["net"]["s_patchout_t_indices"]).astype("str")
            )
            subdir2 += f"_patchout_f_forced_" + removed_bands
        if _config["models"]["net"]["s_patchout_f_interleaved"]:
            subdir2 += f"_patchout_f_interleaved" + str(
                _config["models"]["net"]["s_patchout_f_interleaved"]
            )
        if _config["models"]["net"]["s_patchout_t_interleaved"]:
            subdir2 += f"_patchout_t_interleaved" + str(
                _config["models"]["net"]["s_patchout_t_interleaved"]
            )
        subdir3 = str(_config["inference"]["n_block"])
        out_dir = Path(_config["inference"]["out_dir"]) / subdir1 / subdir2 / subdir3

        for k, v in agg_out.items():
            file_path = out_dir / (k + f".{output}")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(file_path, v)


@ex.command
def extract_embeddings(_run, _config, _log, _rnd, _seed):
    extract_output(_run, _config, _log, _rnd, _seed, output_name="embeddings")


@ex.command
def extract_logits(_run, _config, _log, _rnd, _seed):
    extract_output(_run, _config, _log, _rnd, _seed, output_name="logits")


@ex.command
def test(_config):
    trainer = ex.get_trainer()
    modul = M(ex)
    modul.eval()

    if os.environ["NODE_RANK"] == "0":
        project_name = "discogs_test"
        trainer.logger = CometLogger(
            project_name=project_name,
            api_key=os.environ["COMET_API_KEY"],
        )
        trainer.logger.log_hyperparams(_config)

    trainer.test(modul)


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
        # print(cur_mean, cur_std, np.max(audio_input), np.min(audio_input))
    print(np.mean(mean), np.mean(std))


@ex.command
def test_loaders():
    """
    get one sample from each loader for debbuging
    @return:
    """
    for i, b in enumerate(ex.datasets.training.get_iter()):
        print(b)
        break

    for i, b in enumerate(ex.datasets.test.get_iter()):
        print(b)
        break


@ex.automain
def default_command():
    return main()
