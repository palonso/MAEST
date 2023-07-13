# Music efficient Spectrogram Transformers

- intro

- essentia

- hugging-face

- citing

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
python ex_maest.py extract_embeddings with maest_10s_from_passt_pretrain target_mtt

# Do downstream evaluation
python ex_tl.py with target_mtt
```

## using MAEST in your code
