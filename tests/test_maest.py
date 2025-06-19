import pytest
import numpy as np
import torch
from maest import get_maest


@pytest.fixture
def model():
    # Use a simple config for testing; adjust as needed for your environment
    return get_maest(arch="discogs-maest-30s-pw-129e", pretrained=False)


def test_numpy_input(model):
    input_data = np.random.rand(128, 128)
    with pytest.raises(Exception):
        model(input_data)


def test_empty_input(model):
    input_data = torch.empty([])
    with pytest.raises(Exception):
        model(input_data)


def test_long_2d_input(model):
    # Batch of 2 audio samples of 40 seconds each
    input_data = torch.rand(2, 40 * 16000).float()
    with pytest.raises(Exception):
        model(input_data)


def test_1d_input(model):
    # Patch of 10-second audio
    input_data = torch.rand(10 * 16000).float()
    logits, _ = model(input_data)
    assert logits.shape == (1, 400), "logits shape should be (batch_size, num_classes)"


def test_2d_audio_logits(model):
    input_data = torch.rand(2, 10 * 16000).float()
    # Batch of 2 audio samples
    logits, _ = model(input_data, melspectrogram_input=False)
    assert logits.shape == (2, 400), "Output batch size should match input batch size"


def test_2d_melspec_logits(model):
    # Patch of 30-second mel spectrograms
    input_data = torch.rand(96, 1875).float()
    logits, _ = model(input_data, melspectrogram_input=True)
    assert logits.shape == (1, 400), "Output batch size should match input batch size"


def test_2d_melspec_embeddings(model):
    # Patch of 30-second mel spectrograms
    input_data = torch.rand(96, 1875).float()
    _, embeddings = model(input_data, melspectrogram_input=True, transformer_block=6)
    assert embeddings.shape == (1, 2304), (
        "Output batch size should match input batch size"
    )


def test_3d_melspec_embeddings(model):
    # Batch of 2 mel spectrograms
    input_data = torch.rand(2, 96, 1875).float()
    _, embeddings = model(input_data, melspectrogram_input=True, transformer_block=6)
    assert embeddings.shape == (2, 2304), (
        "Output batch size should match input batch size"
    )


def test_4d_melspec_embeddings(model):
    # Batch of 2 mel spectrograms
    input_data = torch.rand(2, 1, 96, 1875).float()
    _, embeddings = model(input_data, melspectrogram_input=True, transformer_block=6)
    assert embeddings.shape == (2, 2304), (
        "Output batch size should match input batch size"
    )
