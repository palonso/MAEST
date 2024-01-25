import numpy as np
from essentia.pytools.extractors.melspectrogram import melspectrogram
from essentia.standard import MonoLoader


SR = 16000
HOP_SIZE = 256
N_MELS = 96
FRAME_SIZE = 512


def melspectrorgam_extractor(audio_file):
    return melspectrogram(
        audio_file,
        sample_rate=SR,
        frame_size=FRAME_SIZE,
        hop_size=HOP_SIZE,
        window_type='hann',
        low_frequency_bound=0,
        high_frequency_bound=SR / 2,
        number_bands=N_MELS,
        warping_formula='slaneyMel',
        weighting='linear',
        normalize='unit_tri',
        bands_type='power',
        compression_type='shift_scale_log'
    )

melspectrogram_data = melspectrorgam_extractor("techno_loop.wav")
np.save(f"melspectrogram.npy", melspectrogram_data)
