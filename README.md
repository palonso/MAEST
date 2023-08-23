# Music efficient Spectrogram Transformers

This repository contains the code to pre-train, finetune, and infer with the MAEST models.
MAEST is a family of Transformer models based on [PASST](https://github.com/kkoutini/PaSST) and
focused on music analysis applications.

The MAEST models are also available for inference only as part of the
[Essentia](https://essentia.upf.edu/models.html#MAEST) library, and as a [hugging-face models](todo).

# Citing
If you are planning to use MAEST as part of your research, please cite the following paper:

```
    @inproceedings{alonso2023Efficient,
      title={Efficient Supervised Training of Audio Transformers for Music Representation Learning},
      author={Pablo Alonso-Jim{\'e}nez and Xavier Serra and Dmitry Bogdanov},
      booktitle={Proceedings of the International Society for Music Information Retrieval Conference},
      year={2023},
    }
```

# create conda env and activate
    conda create -n PMSST python=3.10
    conda activate PMSST

# install torch 2.0
    pip install numpy --pre torch torchvision torchaudio --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu118

# install rest of stuff
pip install -r requirements.txt

# pre-processing

### Running paper experiments

### pre-training
The code allows to

### Downstream classification
To downstream evaluation is run in two steps: embedding extraction and downstream experiment
involving training and testing.

Run the command `extract_embeddings` from `ex_maest.py` with the pretrained MAEST model and the
target dataset of interest followed by `ex_tl.py`.

```python
# Extract embeddings
python ex_maest.py extract_embeddings with maest_10s_from_passt_inference target_mtt

# Do downstream evaluation
python ex_tl.py with target_mtt_tl
```

## using MAEST in your code
