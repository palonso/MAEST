[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/mtg-upf)
[![arXiv](https://img.shields.io/badge/arXiv-2309.16418-b31b1b.svg)](https://doi.org/10.48550/arXiv.2309.16418)

# Music Audio Efficient Spectrogram Transformer (MAEST)

This repository contains code to pre-train, fine-tune, and infer with the MAEST models.
MAEST is a family of Transformer models based on [PASST](https://github.com/kkoutini/PaSST) and focused on music analysis applications.
Check the [paper](https://doi.org/10.48550/arXiv.2309.16418) for additional details.

The MAEST models are also available for inference from [Essentia](https://essentia.upf.edu/models.html#maest), [Hugging Face](https://huggingface.co/mtg-upf), and [Replicate](https://replicate.com/mtg/maest).

# Install

Our software has been tested in Ubuntu 22.04 LTS, CentOS 7.5, and MacOS Sequoia 15.3.2 using Python 3.10 and 3.12.9.
If MAEST is not working in your current settings we recommend using [Conda](https://docs.conda.io) to set up a working environment with a suitable Python version.

1. (optional) Create a conda environment:

```
conda create -n MAEST python=3.10 -y && conda activate MAEST
```

2. Install MAEST and its dependencies:

```
pip install -e .
```

# Usage

## Using MAEST in your code

MAEST pre-trained models can be loaded in Python both for training and inference:

```python
from maest import get_maest
model = get_maest(arch="discogs-maest-10s-fs-129e")

# Extract logits and embeddings from the last layer
logits, embeddings = model(data)

# Extract embeddings from the 7th layer as reported in the paper.
# This is a vector of 2304 dimensions corresponding to the stack of the CLS, DIST,
# and average of the rest of the tokens.
_, embeddings = model(data, transformer_block=6)
```

MAEST is designed to accept `data` in different input formats:

- 1D: 16kHz audio waveform is assumed.
- 2D: (with `melspectrogram_input=False`) audio is assumed (batch, time).
- 2D: (`melspectrogram_input=True`) mel-spectrogram is assumed (frequency, time).
- 3D: batched mel-spectrogram is assumed (batch, frequency, time).
- 4D: batched mel-spectrgroam plus singleton channel axis is assumed (batch, 1, frequency, time).

On the 1D case, the input audio will be automatically batched.
For the rest of the cases, it is the responsability of the user that the time dimension of the input data is not superior to the maximum length of the model (e.g., 10s, 20s, etc.).
An exception will be raised otherwise.

The models were trained with mel-spectrogram extracted with Essentia's [TensorflowInputMusiCNN](https://essentia.upf.edu/reference/streaming_TensorflowInputMusiCNN.html) algorithm.
However, the inference version uses torch's [torchaudio](https://pytorch.org/audio/stable/index.html) library to extract mel-spectrograms on the GPU.

The following `arch` values are supported:

- `discogs-maest-10s-fs-129e`
- `discogs-maest-10s-pw-129e`
- `discogs-maest-10s-dw-75e`
- `discogs-maest-5s-pw-129e`
- `discogs-maest-20s-pw-129e`
- `discogs-maest-30s-pw-129e`
- `discogs-maest-30s-pw-73e-ts`
- `discogs-maest-30s-pw-129e-519l`

Additionally, `predict_labels()` is an auxiliary function that applies a sigmoid activation, averages the predictions along the time axes, and returns the label vector for convenience.

```python
from maest import get_maest
model = get_maest(arch="discogs-maest-30s-pw-129e-519l")
model.eval()

activations, labels = model.predict_labels(data)
```

## Running the pre-training experiments

We use [Sacred](https://github.com/IDSIA/sacred) to run, configure and log our experiments.
The different routines can be run with Sacred commands, and many experiment options can be directly
configure from the command line.

The output logs are stored in `exp_logs/`, and `exp_logs/lighting_logs/` contains
[TensorBoard](https://www.tensorflow.org/tensorboard) tracking of the experiments.

The following script runs the pre-training routine:

```
python ex_maest.py
```

We provide different Sacred configurations matching the experimental conditions defined in our
paper:

```
# Section 4.2. Impact of initial weights
########################################

# time encodings for up to 10 seconds and initializaiton to random weights
python ex_maest.py with maest_10s_random_weights_pretrain

# time encodings for up to 10 seconds and initializaiton to the DeiT weights
python ex_maest.py with maest_10s_from_deit_pretrain

# time encodings for up to 10 seconds and initializaiton to the PaSST weights
python ex_maest.py with maest_10s_from_passt_pretrain


# Section 4.3. Effect of the input sequence length
##################################################

# time encodings for up to 5 seconds and initializaiton to the PaSST weights
python ex_maest.py with maest_5s_from_passt_pretrain

# time encodings for up to 20 seconds and initializaiton to the PaSST weights
python ex_maest.py with maest_20s_from_passt_pretrain

# time encodings for up to 30 seconds and initializaiton to the PaSST weights
python ex_maest.py with maest_30s_from_passt_pretrain


# Teacher student setup (requires extracting logits from a pretrained model)
############################################################################

python ex_maest.py with maest_30s_teacher_student_pretrain
```

### Pre-training data

Due to copyright limitations, we don't share our pre-training dataset (Discogs20) in this
repository.
To generate your custom pre-training dataset:

1. Pre-extract mel-spectrograms (or your favourite representation) for the dataset. As an example, check the [MagnaTagATune's pre-processing](datasets/mtt/preprocess.py).

2. Generate groundtruth files. We use binary pickle files that store the ground truth as a dictionary
   `"path" : (labels tuple)`. Check the MagnaTagATune training ground truth [file](datasets/mtt/groundtruth-train.pk) as an example.

3. Update the configuration related to the ground truth files. For example:

```bash
python ex_maest.py maest_discogos_30s_pretrain with \
    datamodule.groundtruth_train=my_dataset/groundtruth-train.pk \
    datamodule.groundtruth_val=my_dataset/groundtruth-val.pk \
    datamodule.groundtruth_test=my_dataset/groundtruth-test.pk \
    datamodule.base_dir=my/data/dir/
```

In case more details are required, do not hesitate to contact us for any further question or clarification.

## Inference

We provide a number of options to extract embeddings from the pre-trained MAEST models presented in
the paper:

- `extract_embeddings`: returns a [3, 768] vector with the embeddings for each audio file in
  the `predict` dataset. The three dimensions of the first axis correspond to the CLS token, the
  DIST token and the average of the rest of tokens (see Section 4.1 from the paper).
- `extract_logits`: returns logits that can be used in the teacher student setup, or transformed into label predictions by applying a `Sigmoid` function.

Each pre-training configuration has its inference version. For example, to extract embeddings
with MAEST trained on 10s patches with random weight initialization do:

```
python ex_maest.py extract_embeddings with maest_10s_random_weights_inference
```

The transformer block used to extract the embeddings can be configured as follows:

```
python ex_maest.py extract_embeddings with maest_10s_random_weights_inference predict.transformer_block=11
```

## Downstream evaluation

The downstream evaluation requires the following steps:

1. Dataset pre-processing. For example, for the MagnaTagATune:

```bash
cd datasets/mtt/ && python preprocess.py && cd ../../
```

2. Embedding extraction:

```bash
python ex_maest.py extract_embeddings with maest_30s_from_passt_inference target_mtt
```

3. Downstream experiment running involving training and testing:

```bash
python ex_tl.py with target_mtt_tl
```

# Citing

If you are going to use MAEST as part of your research, please consider citing the following work:

```
    @inproceedings{alonso2023Efficient,
      title={Efficient Supervised Training of Audio Transformers for Music Representation Learning},
      author={Pablo Alonso-Jim{\'e}nez and Xavier Serra and Dmitry Bogdanov},
      booktitle={Proceedings of the International Society for Music Information Retrieval Conference},
      year={2023},
    }
```
