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


def test_1d_input(model):
    input_data = torch.rand(10 * 16000).float()
    activations, _ = model(input_data)
    assert activations.shape == (1, 400), (
        "Activations shape should be (batch_size, num_classes)"
    )


def test_2d_audio_activations(model):
    input_data = torch.rand(2, 10 * 16000).float()
    # Batch of 2 audio samples
    activations, _ = model(input_data, melspectrogram_input=False)
    assert activations.shape == (2, 400), (
        "Output batch size should match input batch size"
    )


def test_2d_melspec_activations(model):
    input_data = torch.rand(96, 1875).float()
    # Batch of 10 mel spectrograms
    activations, _ = model(input_data, melspectrogram_input=True)
    assert activations.shape == (1, 400), (
        "Output batch size should match input batch size"
    )


def test_2d_melspec_embeddings(model):
    input_data = torch.rand(96, 1875).float()
    # Batch of 10 mel spectrograms
    _, embeddings = model(input_data, melspectrogram_input=True, transformer_block=6)
    assert embeddings.shape == (1, 2304), (
        "Output batch size should match input batch size"
    )
