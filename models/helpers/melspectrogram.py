import torch
from torch.nn import Module
from torchaudio.transforms import Spectrogram, MelScale

# According to Pytoch, mel-spectrogram should be implemented as a module:
# https://pytorch.org/audio/stable/transforms.html

# WARNING: The torchaudio implementation is similar but not identical to Essentia's.
# Relative tolerance is 1e-3 and absolute tolerance is 1e-3. We assume that this has minimal
# impact in the resulting embeddings.


class MelSpectrogram(Module):
    """Extract mel-spectgrams as a torchaudio module"""

    sr = 16000
    win_len = 512
    hop_len = 256
    power = 2
    n_mel = 96
    norm = "slaney"
    mel_scale_type = "slaney"
    norm_mean = 2.06755686098554
    norm_std = 1.268292820667291

    def __init__(self):
        super().__init__()

        self.spec = Spectrogram(
            n_fft=self.win_len,
            win_length=self.win_len,
            hop_length=self.hop_len,
            power=self.power,
        )

        self.mel_scale = MelScale(
            n_mels=self.n_mel,
            sample_rate=self.sr,
            n_stft=self.win_len // 2 + 1,
            norm=self.norm,
            mel_scale=self.mel_scale_type,
        )

    def znorm(self, input_values: torch.Tensor) -> torch.Tensor:
        return (input_values - (self.norm_mean)) / (self.norm_std * 2)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # convert to power spectrogram
        spec = self.spec(waveform)

        # convert to mel-scale
        mel = self.mel_scale(spec)

        # apply logC compression
        logmel = torch.log10(1 + mel * 10000)

        # normalize
        logmel = self.znorm(logmel)

        return logmel
