# Music Audio  Efficient Spectrogram Transformer (MAEST)

This repository contains code to pre-train, finetune, and infer with the MAEST models.
MAEST is a family of Transformer models based on [PASST](https://github.com/kkoutini/PaSST) and
focused on music analysis applications.

The MAEST models are also available for inference only as part of the
[Essentia](https://essentia.upf.edu/models.html#MAEST) library, and as a [hugging-face models](todo).

# Install 

We recommend using the [Conda](https://docs.conda.io) package manager to setup the working envoronment. 


1. Create a conda environment:

```
conda create -n MAEST python=3.10
conda activate MAEST
```

2. Install torch 2.0:

```
pip install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118
```

3. Install rest of packages:

```
pip install -r requirements.txt
```

# Usage

We use [Sacred]() to run, configure an log our experiments. 
The different routines can be run with Sacred commands, and many experiment options can be diretly
configure from the command line.

## Running the pre-training experiments

The following script will run the pre-training routine.

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
python ex_maest.py with maest_discogs_5sec_pretrain

# time encodings for up to 10 seconds and initializaiton to the PaSST weights
python ex_maest.py with maest_discogs_10sec_pretrain

# time encodings for up to 20 seconds and initializaiton to the PaSST weights
python ex_maest.py with maest_discogs_20sec_pretrain

# time encodings for up to 30 seconds and initializaiton to the PaSST weights
python ex_maest.py with maest_discogs_30sec_pretrain


# Teacher student setup (requires extracting logits from a pretrained model)
############################################################################

python ex_maest.py with maest_30s_teacher_student_pretrain
```

### Pre-training data

Due to copyright limitations, we don't share our pre-training dataset (Discogs20) in this
repository.
To generate a new pre-training dataset:

1. Pre-extract mel-spectrogram representation for the input dataset similar to the process done in
   the [MagnaTagATune example](datasets/mtt/preprocess.py).

2. Generate groundtruth files. We use binary pickle files that store the groundtruth as a dictionary
   `"path" : (labels tuple)`. Check the [MagnaTagATune training groundtruth file](dataset/mtt/groundtruth-train.pk) as an example.

3. Update the configuration related to the groundtruth files. For example:

```bash
python ex_maest.py maest_discogos_30s_pretrain with \
    datamodule.groundtruth_train=my_dataset/groundtruth-train.pk \
    datamodule.groundtruth_val=my_dataset/groundtruth-val.pk \
    datamodule.groundtruth_test=my_dataset/groundtruth-test.pk \
    datamodule.base_dir =my/data/dir/
```

Do not hesitate to  contact us for any further question or clarification related to the MAEST pre-training.

## Inference

We provide a number of option to extract embeddings from the pre-trained models:

- `extract_embeddings`: which retuns a [3, 768] vector with the embeddings for each audio file in
  the `predict` dataset. The three dimensions of the first axis correspond to the CLS token, the
  DIST token and the average of the rest of tokensi (see Section 4.1 from the paper). 
- `extract_logits`: that can be used in a teacher student setup, or transformed into label
  predictions by applying a `Sigmoid` function.

Each pre-training configuration has its inference correspondence. For example, to extract embeddings
with MAEST trained on 10s patches with random weights initialization do:

```
python ex_maest.py extract_embeddings with maest_10s_random_weights_inference
```

The transformer block used to extract the embeddings can be configured as follows:

```
python ex_maest.py extract_embeddings with maest_10s_random_weights_inference inference.transformer_block=11
```

## Downstream evaluation

The downstream evaluation required thw following steps:

1. Dataset pre-processing. For example, for the MagnaTagATune:

```bash
cd dataset/mtt/ && python preprocess.py && cd ../../
```

2. Embedding extraction:

```bash
python ex_maest.py extract_embeddings with maest_30s_from_passt_infer target_mtt
```

3. Downstream experiment running involving training and testing:

```bash
python ex_tl.py with target_mtt_tl
```


# Citing
If you are planning to use MAEST as part of your research, please cite the following work:

```
    @inproceedings{alonso2023Efficient,
      title={Efficient Supervised Training of Audio Transformers for Music Representation Learning},
      author={Pablo Alonso-Jim{\'e}nez and Xavier Serra and Dmitry Bogdanov},
      booktitle={Proceedings of the International Society for Music Information Retrieval Conference},
      year={2023},
    }
```

