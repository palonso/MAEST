from argparse import ArgumentParser
import json
from models.maest import maest
from essentia.standard import MonoLoader
from scipy.special import expit
import numpy as np
from torch import Tensor

parser = ArgumentParser()
parser.add_argument("--audio", action="store_true")
args = parser.parse_args()
use_audio = args.audio

# metadata_file = "discogs-effnet-bs64-1.json"
# with open(metadata_file) as f:
#    metadata = json.load(f)
#    classes = metadata["classes"]

timestamps = 1875
audio_file = "a1RR7RcamlU.mp4"
melspec_file = "/data0/palonso/data/discotube30s/a1/a1RR7RcamlU.mp4.mmap"

if use_audio:
    data = MonoLoader(filename=audio_file, sampleRate=16000, resampleQuality=4)()
else:
    melspec_data = np.memmap(melspec_file, dtype=np.float16, mode="r")
    melspec = np.array(melspec_data).reshape(-1, 96)
    mean = 2.06755686098554
    std = 1.268292820667291
    melspec = (melspec - mean) / (std * 2)
    trim = melspec.shape[0] % timestamps
    if trim:
        melspec = melspec[:-trim, :]

    melspec = melspec.T
    print(np.info(melspec))
    data = melspec.reshape(-1, 1, 96, timestamps)
    data = Tensor(data)

models = (
    ("discogs-maest-10s-fs-129e", 625),
    ("discogs-maest-10s-pw-129e", 625),
    ("discogs-maest-10s-dw-75e", 625),
    ("discogs-maest-5s-pw-129e", 312),
    ("discogs-maest-20s-pw-129e", 1250),
    ("discogs-maest-30s-pw-129e", 1875),
    ("discogs-maest-30s-pw-73e-ts", 1875),
)
for model, timestamps in models:
    print(f"processing {model} with {timestamps} timestamps")
    model = maest(arch=model)

    activations, labels = model.predict_labels(data)

    for i, l in enumerate(activations.argsort()[-5:][::-1], 1):
        print('{}: {} ({:.2f}%)'.format(i, labels[l], activations[l] * 100))
