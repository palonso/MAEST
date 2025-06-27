import torch
from maest import get_maest
from essentia.standard import MonoLoader

audio = MonoLoader(filename="/Users/palonso/rock.mp3", sampleRate=16000)()

model = get_maest(arch="discogs-maest-30s-pw-129e")

# t = int(16000 * 31)
# audio = torch.randn(4, t)
x, embeddings = model(torch.tensor(audio[: 16000 * 10]))

print("audio shape", x.shape)

arg_max = torch.argmax(x, dim=1)
print("activations arg max", arg_max)
