import os
import pickle
import logging

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    WeightedRandomSampler,
)
from sacred import Ingredient

from helpers.audiodatasets import PreprocessDataset
from helpers.spec_masking import SpecMasking
from .dataset import DiscogsDataset, DiscogsDatasetExhaustive


datamodule_ing = Ingredient("datamodule")
_logger = logging.getLogger("datamodule")


@datamodule_ing.config
def default_config():
    base_dir = "/data0/palonso/data/discotube30s/"
    base_dir_val = ""  # alternative location for the validation data
    groundtruth_train = "discogs/gt_train_all_400l_super_clean.pk"
    groundtruth_val = "discogs/gt_val_all_400l_super_clean.pk"
    groundtruth_test = "discogs/gt_test_all_400l_super_clean.pk"
    # groundtruth_predict = "discogs/gt_train_all_400l_super_clean.pk"
    groundtruth_predict = "discogs/gt_val_all_400l_super_clean.pk"

    batch_size_train = 12
    batch_size_test = 20
    num_workers = 16

    clip_length = 10

    roll = {
        "do": False,  # apply roll augmentation
        "axis": -1,
        "shift": None,
        "shift_range": 50,
    }

    norm = {
        "do": True,  # normalize dataset
        "norm_mean": 2.06755686098554,
        "norm_std": 1.268292820667291,
    }

    masking = {
        "do": True,
        "time_mask_param": 8,
        "freq_mask_param": 5,
        "p": 0.2,
        "iid_masks": True,
        "time_masks": 20,
        "freq_masks": 8,
    }

    sampler = {
        "sample_weight_offset": 100,
        "sample_weight_sum": True,
        "sampler_replace": False,
        "epoch_len": 200000,
    }

    teacher_student = {
        "teacher_target": False,
        "teacher_target_base_dir": "",
        "teacher_target_threshold": 0.45,
    }


class DistributedSamplerWrapper(DistributedSampler):
    def __init__(
        self, sampler, dataset, num_replicas=None, rank=None, shuffle: bool = True
    ):
        super(DistributedSamplerWrapper, self).__init__(
            dataset, num_replicas, rank, shuffle
        )
        # source: @awaelchli https://github.com/PyTorchLightning/pytorch-lightning/issues/3238
        self.sampler = sampler

    def __iter__(self):
        if self.sampler.generator is None:
            self.sampler.generator = torch.Generator()
        self.sampler.generator.manual_seed(self.seed + self.epoch)
        indices = list(self.sampler)
        if self.epoch == 0:
            _logger.info(f"DistributedSamplerWrapper: {indices[:10]}")
        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)


class DiscogsDataModule(pl.LightningDataModule):
    @datamodule_ing.capture
    def __init__(self, masking):
        super().__init__()

        if masking["do"]:
            params = {k: v for k, v in masking.items() if k != "do"}
            self.spec_masking = SpecMasking(**params)

    @datamodule_ing.capture(prefix="roll")
    def get_roll_func(self, axis, shift, shift_range):
        _logger.info("rolling...")

        def roll_func(b):
            x, f, y = b
            x = torch.as_tensor(x)
            sf = shift
            if shift is None:
                sf = int(np.random.random_integers(-shift_range, shift_range))

            return x.roll(sf, axis), f, y

        return roll_func

    @datamodule_ing.capture(prefix="norm")
    def get_norm_func(self, norm_mean, norm_std):
        _logger.info("normalizing...")

        def norm_func(b):
            x, f, y = b
            x = (x - norm_mean) / (norm_std * 2)

            return x, f, y

        return norm_func

    def get_masking_func(self):
        _logger.info("masking...")

        def masking_func(b):
            x, f, y = b
            x = torch.as_tensor(x)
            self.spec_masking.compute(x)

            return x, f, y

        return masking_func

    @datamodule_ing.capture(prefix="sampler")
    def get_ft_cls_balanced_sample_weights(
        self, groundtruth, sample_weight_offset, sample_weight_sum
    ):
        """
        :return: float tenosr of shape len(full_training_set) representing the weights of each sample.
        """
        with open(groundtruth, "rb") as dataset_file:
            dataset_data = pickle.load(dataset_file)
            all_y = np.array(list(dataset_data.values()))

        all_y = torch.as_tensor(all_y)
        per_class = all_y.long().sum(0).float().reshape(1, -1)  # frequencies per class
        per_class += sample_weight_offset  # offset low freq classes

        if sample_weight_offset > 0:
            _logger.info(
                f"Warning: sample_weight_offset={sample_weight_offset} minnow={per_class.min()}"
            )

        per_class_weights = 1000.0 / per_class
        all_weight = all_y * per_class_weights
        if sample_weight_sum:
            _logger.info("sample_weight_sum")
            all_weight = all_weight.sum(dim=1)
        else:
            all_weight, _ = all_weight.max(dim=1)
        return all_weight

    @datamodule_ing.capture(prefix="sampler")
    def get_ft_weighted_sampler(
        self,
        groundtruth,
        epoch_len,
        sampler_replace,
    ):
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if world_size > 1:
            _logger.info(f"WORLD_SIZE: {world_size}")
            _logger.info(f"LOCAL_RANK: {local_rank}")

        sample_weights = self.get_ft_cls_balanced_sample_weights(
            groundtruth=groundtruth
        )

        return DistributedSamplerWrapper(
            sampler=WeightedRandomSampler(
                sample_weights, num_samples=epoch_len, replacement=sampler_replace
            ),
            dataset=range(epoch_len),
            num_replicas=world_size,
            rank=local_rank,
        )

    @datamodule_ing.capture
    def train_dataloader(
        self,
        groundtruth_train,
        base_dir,
        clip_length,
        batch_size_train,
        num_workers,
        norm,
        roll,
        masking,
    ):
        ds = DiscogsDataset(
            groundtruth_train,
            base_dir=base_dir,
            clip_length=clip_length,
        )

        if norm["do"]:
            ds = PreprocessDataset(ds, self.get_norm_func())
        if roll["do"]:
            ds = PreprocessDataset(ds, self.get_roll_func())
        if masking["do"]:
            ds = PreprocessDataset(ds, self.get_masking_func())

        return DataLoader(
            dataset=ds,
            sampler=self.get_ft_weighted_sampler(groundtruth=groundtruth_train),
            batch_size=batch_size_train,
            num_workers=num_workers,
            shuffle=None,
        )

    @datamodule_ing.capture
    def val_dataloader(
        self,
        groundtruth_val,
        base_dir,
        base_dir_val,
        clip_length,
        batch_size_test,
        num_workers,
        norm,
    ):
        if not base_dir_val:
            base_dir_val = base_dir

        ds = DiscogsDataset(
            groundtruth_val,
            base_dir_val,
            clip_length=clip_length,
        )

        if norm["do"]:
            ds = PreprocessDataset(ds, self.get_norm_func())

        return DataLoader(
            dataset=ds,
            batch_size=batch_size_test,
            num_workers=num_workers,
        )

    @datamodule_ing.capture
    def test_dataloader(
        self,
        groundtruth_test,
        base_dir,
        clip_length,
        batch_size_test,
        num_workers,
        norm,
    ):
        ds = DiscogsDatasetExhaustive(
            groundtruth_test, base_dir, clip_length=clip_length
        )

        if norm["do"]:
            ds = PreprocessDataset(ds, self.get_norm_func())

        return DataLoader(
            dataset=ds,
            batch_size=batch_size_test,
            num_workers=num_workers,
        )

    @datamodule_ing.capture
    def predict_dataloader(
        self,
        groundtruth_predict,
        base_dir,
        clip_length,
        batch_size_test,
        num_workers,
        norm,
    ):
        ds = DiscogsDatasetExhaustive(
            groundtruth_predict, base_dir, clip_length=clip_length
        )

        if norm["do"]:
            ds = PreprocessDataset(ds, self.get_norm_func())

        return DataLoader(
            dataset=ds,
            batch_size=batch_size_test,
            num_workers=num_workers,
        )
