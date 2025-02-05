import numpy as np
from essentia.streaming import (
    VectorInput,
    MonoLoader,
    FrameCutter,
    TensorflowInputMusiCNN,
)
from essentia import Pool, run, reset


class MelSpectrogramExtractor:
    # mel-spec signature params
    sample_rate = 16000
    resample_quality = 4
    frame_size = 512
    hop_size = 256

    def __init__(self):
        self.pool = Pool()
        self.frame_cutter = FrameCutter(
            frameSize=self.frame_size,
            hopSize=self.hop_size,
            silentFrames="keep",
        )
        self.melspec = TensorflowInputMusiCNN()

        self.frame_cutter.frame >> self.melspec.frame
        self.melspec.bands >> (self.pool, "mel_bands")

    def __call__(self, audio):
        # make sure audio is float32 for Essentia
        audio = audio.astype(np.float32)

        # set vector input and connect the network
        vector_input = VectorInput(audio)
        vector_input.data >> self.frame_cutter.signal

        # run the network and get the mel bands
        run(vector_input)
        mel_bands = np.array(self.pool["mel_bands"])

        # clear the pool and reset the vector input
        reset(vector_input)
        vector_input.data.disconnect(self.frame_cutter.signal)
        self.pool.clear()
        del vector_input

        # return freq, time
        return mel_bands.T

    def load_audio(self, audio_file):
        return MonoLoader(
            filename=audio_file,
            sampleRate=self.sample_rate,
            resampleQuality=self.resample_quality,
        )()
