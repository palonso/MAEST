import pathlib
import pickle
import random
import logging

import numpy as np
from torch.utils.data import Dataset as TorchDataset
from sacred import Ingredient
from scipy.special import expit

dataset_ing = Ingredient("dataset")
_logger = logging.getLogger("dataset")


@dataset_ing.config
def default_config():
    name = "discogs"  # dataset name

    sample_rate = 16000
    hop_size = 256
    n_bands = 96

    half_overlapped_inference = False


class DiscogsDataset(TorchDataset):
    @dataset_ing.capture
    def __init__(
        self,
        groundtruth_file,
        base_dir,
        sample_rate,
        clip_length,
        hop_size,
        n_bands,
    ):
        """
        Reads the mel spectrogram chunks with numpy and returns a fixed length mel-spectrogram patch
        """

        self.base_dir = base_dir
        with open(groundtruth_file, "rb") as gf:
            self.groundtruth = pickle.load(gf)
        self.filenames = {
            i: filename for i, filename in enumerate(list(self.groundtruth.keys()))
        }
        self.length = len(self.groundtruth)
        self.sample_rate = sample_rate
        self.dataset_file = None  # lazy init
        self.clip_length = clip_length

        self.melspectrogram_size = clip_length * sample_rate // hop_size
        self.n_bands = n_bands

    def __len__(self):
        return self.length

    def __del__(self):
        if self.dataset_file is not None:
            self.dataset_file.close()
            self.dataset_file = None

    def __getitem__(self, index):
        """Load waveform and target of an audio clip."""

        filename = self.filenames[index]
        target = self.groundtruth[filename].astype("float16")

        melspectrogram_file = pathlib.Path(self.base_dir, filename)
        melspectrogram = self.load_melspectrogram(melspectrogram_file)

        return melspectrogram, str(filename), target

    def load_melspectrogram(
        self, melspectrogram_file: pathlib.Path, offset: int = None
    ):
        if melspectrogram_file.suffix == ".npy":
            melspectrogram = np.load(melspectrogram_file).astype("float16")

            if melspectrogram.shape[0] < self.melspectrogram_size:
                padding_size = self.melspectrogram_size - melspectrogram.shape[0]
                melspectrogram = np.vstack(
                    [
                        melspectrogram,
                        np.zeros([padding_size, self.n_bands], dtype="float16"),
                    ]
                )
                melspectrogram = np.roll(
                    melspectrogram, padding_size // 2, axis=0
                )  # center the padding
            else:
                melspectrogram = melspectrogram[: self.melspectrogram_size, :]

        else:
            frames_num = melspectrogram_file.stat().st_size // (
                2 * self.n_bands
            )  # each float16 has 2 bytes

            if type(offset) is not int:
                max_frame = frames_num - self.melspectrogram_size
                offset = random.randint(0, max(max_frame, 0))

            # offset: idx * bands * bytes per float
            offset_bytes = offset * self.n_bands * 2

            skip_frames = max(offset + self.melspectrogram_size - frames_num, 0)
            frames_to_read = self.melspectrogram_size - skip_frames

            try:
                fp = np.memmap(
                    melspectrogram_file,
                    dtype="float16",
                    mode="r",
                    shape=(frames_to_read, self.n_bands),
                    offset=offset_bytes,
                )
            except Exception:
                _logger.error(f"Error loading {melspectrogram_file}")
                _logger.error(
                    f"num frames: {frames_num}, offset: {offset}, skip frames: {skip_frames}, frames to read: {frames_to_read}"
                )
                raise

            # put the data in a numpy ndarray
            melspectrogram = np.array(fp, dtype="float16")

            if frames_to_read < self.melspectrogram_size:
                padding_size = self.melspectrogram_size - frames_to_read
                melspectrogram = np.vstack(
                    [
                        melspectrogram,
                        np.zeros([padding_size, self.n_bands], dtype="float16"),
                    ]
                )
                melspectrogram = np.roll(
                    melspectrogram, padding_size // 2, axis=0
                )  # center the padding

            del fp

        # transpose, PaSST expects dims as [b,e,f,t]
        melspectrogram = melspectrogram.T
        melspectrogram = np.expand_dims(melspectrogram, 0)

        return melspectrogram


class DiscogsDatasetTS(DiscogsDataset):
    @dataset_ing.capture
    def __init__(
        self,
        groundtruth_file,
        base_dir,
        sample_rate,
        clip_length,
        hop_size,
        n_bands,
        teacher_target_base_dir,
        teacher_target_threshold,
    ):
        super().__init__(
            groundtruth_file,
            base_dir=base_dir,
            sample_rate=sample_rate,
            clip_length=clip_length,
            hop_size=hop_size,
            n_bands=n_bands,
        )

        self.teacher_target_base_dir = teacher_target_base_dir
        self.teacher_target_threshold = teacher_target_threshold

    def __getitem__(self, index):
        """Load waveform and target of an audio clip."""

        filename = self.filenames[index]
        target = self.groundtruth[filename].astype("float16")

        melspectrogram_file = pathlib.Path(self.base_dir, filename)
        melspectrogram = self.load_melspectrogram(melspectrogram_file)

        teacher_target_file = pathlib.Path(
            self.teacher_target_base_dir, str(filename) + ".logits.npy"
        )
        teacher_target = np.load(teacher_target_file).astype("float16").squeeze()

        teacher_target = expit(teacher_target)

        # hard teacher target with a threshold of 0.45
        hard_teacher_target = (teacher_target > self.teacher_target_threshold).astype(
            "float16"
        )
        # if no class is activated, set the highest activation
        if not np.sum(hard_teacher_target):
            hard_teacher_target = np.zeros(hard_teacher_target.shape, dtype="float16")
            hard_teacher_target[np.argmax(teacher_target)] = 1.0

        return melspectrogram, str(filename), target, hard_teacher_target


class DiscogsDatasetExhaustive(DiscogsDataset):
    @dataset_ing.capture
    def __init__(
        self,
        groundtruth_file,
        base_dir,
        sample_rate,
        clip_length,
        hop_size,
        n_bands,
        half_overlapped_inference,
    ):
        """
        Reads the mel spectrogram chunks with numpy and returns a fixed length mel-spectrogram patch
        """
        super().__init__(
            groundtruth_file,
            base_dir=base_dir,
            sample_rate=sample_rate,
            clip_length=clip_length,
            hop_size=hop_size,
            n_bands=n_bands,
        )
        self.hop_size = (
            self.melspectrogram_size // 2
            if half_overlapped_inference
            else self.melspectrogram_size
        )
        self.half_overlap = half_overlapped_inference

        if pathlib.Path(list(self.filenames.values())[0]).suffix == ".mmap":
            filenames = []
            for filename in self.filenames.values():
                melspectrogram_file = pathlib.Path(self.base_dir, filename)
                frames_num = melspectrogram_file.stat().st_size // (
                    2 * self.n_bands
                )  # each float16 has 2 bytes
                if self.half_overlap:
                    frames_num -= self.hop_size

                # allow 10% margin with zero-pad
                n_patches = int((frames_num * 1.1) // self.hop_size)
                # filenames is a tuple (filename, offset)
                filenames.extend(
                    [(filename, i * self.hop_size) for i in range(n_patches)]
                )
        else:
            filenames = list(zip(self.filenames.values(), [0] * len(self.filenames)))

        self.filenames_with_patch = dict(zip(range(len(filenames)), filenames))
        self.length = len(self.filenames_with_patch)

    def __getitem__(self, index):
        """Load waveform and target of an audio clip."""

        filename, offset = self.filenames_with_patch[index]
        target = self.groundtruth[filename].astype("float16")

        melspectrogram_file = pathlib.Path(self.base_dir, filename)
        melspectrogram = self.load_melspectrogram(melspectrogram_file, offset)

        return melspectrogram, str(filename), target


class DiscogsDatasetExhaustiveTS(DiscogsDatasetExhaustive):
    @dataset_ing.capture
    def __init__(
        self,
        groundtruth_file,
        base_dir,
        sample_rate,
        clip_length,
        hop_size,
        n_bands,
        half_overlap,
        teacher_target,
        teacher_target_base_dir,
        teacher_target_threshold,
    ):
        """
        Reads the mel spectrogram chunks with numpy and returns a fixed length mel-spectrogram patch
        """
        super().__init__(
            groundtruth_file,
            base_dir=base_dir,
            sample_rate=sample_rate,
            clip_length=clip_length,
            hop_size=hop_size,
            n_bands=n_bands,
            half_overlap=half_overlap,
        )

        self.teacher_target = teacher_target
        self.teacher_target_base_dir = teacher_target_base_dir
        self.teacher_target_threshold = teacher_target_threshold

    def __getitem__(self, index):
        """Load waveform and target of an audio clip."""

        filename, offset = self.filenames_with_patch[index]
        target = self.groundtruth[filename].astype("float16")

        melspectrogram_file = pathlib.Path(self.base_dir, filename)
        melspectrogram = self.load_melspectrogram(melspectrogram_file, offset)

        teacher_target_file = pathlib.Path(
            self.teacher_target_base_dir, filename + ".logits.npy"
        )
        teacher_target = np.load(teacher_target_file).astype("float16").squeeze()

        # logits to activations
        teacher_target = expit(teacher_target)

        # hard teacher target with a threshold of 0.45
        hard_teacher_target = (teacher_target > self.teacher_target_threshold).astype(
            "float16"
        )
        # if no class is activated, set the highest activation
        if not np.sum(hard_teacher_target):
            hard_teacher_target = np.zeros(hard_teacher_target.shape, dtype="float16")
            hard_teacher_target[np.argmax(teacher_target)] = 1.0

        return melspectrogram, str(filename), target, hard_teacher_target


@dataset_ing.capture
def print_conf(_config):
    _logger.info(f"Config of {dataset_ing.path}, with id: {id(dataset_ing)}")
    _logger.info(_config)
