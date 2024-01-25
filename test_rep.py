import numpy as np
from transformers import ASTFeatureExtractor
from essentia.standard import MonoLoader

from models.discogs_labels import discogs_labels

audio_file = "a1RR7RcamlU.mp4"
melspec_file = "/data0/palonso/data/discotube30s/a1/a1RR7RcamlU.mp4.mmap"
timestamps = 626


# reference
melspec_data = np.memmap(melspec_file, dtype=np.float16, mode="r")
expected = np.array(melspec_data).reshape(-1, 96)

audio = MonoLoader(filename=audio_file, sampleRate=16000, resampleQuality=4)()

mean = 2.06755686098554
std = 1.268292820667291
# melspec_normalization = (melspec - mean) / (std * 2)


extractor = ASTFeatureExtractor(
    sample_rate=16000,
    num_mel_bins=96,
    do_normalize=False,
)

found = extractor(audio, sampling_rate=16000, max_length=None, return_tensors="np")['input_values']

print("expected shape:", expected.shape)
print("expected max:", expected.max())
print("expected min:", expected.min())
print("found shape:", found.shape)
print("found max:", found.max())
print("found min:", found.min())
