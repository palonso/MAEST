import argparse
import os
from pathlib import Path

import numpy as np
from essentia.streaming import (
    VectorInput,
    MonoLoader,
    FrameCutter,
    Windowing,
    Spectrum,
    MelBands,
    UnaryOperator,
)
from essentia import Pool, run, reset


class MelSpectrogramExtractor:
    sample_rate = 16000
    resample_quality = 4
    hop_size = 256
    frame_size = 512
    n_mels = 96
    window_type = 'hann'
    low_frequency_bound = 0
    high_frequency_bound = sample_rate / 2
    warping_formula = 'slaneyMel'
    weighting = 'linear'
    normalize = 'unit_tri'
    bands_type = 'power'

    def __init__(self):
        self.pool = Pool()
        self.frameCutter = FrameCutter(
            frameSize=self.frame_size,
            hopSize=self.hop_size,
            silentFrames="keep",
        )
        self.windowing = Windowing(
            type=self.window_type,
            normalized=False,
        )
        self.spec = Spectrum(size=self.frame_size)
        self.mels = MelBands(
            inputSize=self.frame_size // 2 + 1,
            numberBands=self.n_mels,
            sampleRate=self.sample_rate,
            lowFrequencyBound=self.low_frequency_bound,
            highFrequencyBound=self.high_frequency_bound,
            warpingFormula=self.warping_formula,
            weighting=self.weighting,
            normalize=self.normalize,
            type=self.bands_type,
            log=False,
        )

        self.shift = UnaryOperator(type='identity', scale=1e4, shift=1)
        self.compressor = UnaryOperator(type='log10')

        self.frameCutter.frame >> self.windowing.frame >> self.spec.frame
        self.spec.spectrum >> self.mels.spectrum
        self.mels.bands >> self.shift.array >> self.compressor.array >> (self.pool, 'mel_bands')

    def __call__(self, audio):
        # make sure audio is float32 for Essentia
        audio = audio.astype(np.float32)

        # set vector input and connect the network
        vector_input = VectorInput(audio)
        vector_input.data >> self.frameCutter.signal

        run(vector_input)
        mel_bands = np.array(self.pool['mel_bands'])

        self.pool.clear()
        reset(vector_input)

        vector_input.data.disconnect(self.frameCutter.signal)
        del vector_input

        # to freq, time
        return mel_bands.T

    def load_audio(self, audio_file):
        return MonoLoader(
            filename=audio_file,
            sampleRate=self.sample_rate,
            resampleQuality=self.resample_quality
        )()

