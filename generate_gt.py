import torch
from models.maest import maest
from transformers import pipeline
import numpy as np
from essentia.standard import MonoLoader
from essentia.standard import TensorflowPredict
from essentia import Pool

from models.discogs_labels import discogs_labels

org = "mtg-upf"

models = [
    # "discogs-maest-10s-fs-129e",
    # "discogs-maest-10s-pw-129e",
    # "discogs-maest-10s-dw-75e",
    # "discogs-maest-5s-pw-129e",
    # "discogs-maest-20s-pw-129e",
    "discogs-maest-30s-pw-129e",
    # "discogs-maest-30s-pw-73e-ts",
]


audio = MonoLoader(filename="/home/palonso/techno_loop.wav", sampleRate=16000, resampleQuality=4)()
audio = audio[:16000 * 30]


for model in models:
    model_maest = maest(model)
    logits_maest, _ = model_maest(audio)
    print(f"logits_maest shape: {logits_maest.shape}")

    # preds_ast = torch.mean(torch.sigmoid(logits_ast), dim=0)
    preds_maest = torch.mean(torch.sigmoid(logits_maest), dim=0)
    np.save(f"preds_maest_{model}.npy", preds_maest.detach().numpy())
