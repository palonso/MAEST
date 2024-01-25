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


#  audio = MonoLoader(filename="/home/palonso/rock.mp3", sampleRate=16000, resampleQuality=4)()
#  audio = audio[:16000 * 30]

timestamps = 1876
melspec_file = "/data0/palonso/data/discotube30s/a1/a1RR7RcamlU.mp4.mmap"


melspec_data = np.memmap(melspec_file, dtype=np.float16, mode="r")
melspec = np.array(melspec_data).reshape(-1, 96)

melspec = np.load("melspectrogram.npy")
melspec = melspec[:timestamps, :]
mean = 2.06755686098554
std = 1.268292820667291
melspec = (melspec - mean) / (std * 2)
trim = melspec.shape[0] % timestamps
if trim:
    melspec = melspec[:trim, :]

melspec = melspec.T
# melspec = np.append(melspec, np.zeros([96, 1]), axis=-1)
print(f"melspec shape: {melspec.shape}")
melspec = melspec.reshape(96, -1, timestamps)
# cut into equally-sized patches.
# Note that the new batch axis needs to be next to the time.
# resort axes: batch, channels, freq, time
melspec = np.swapaxes(melspec, 0, 1)
data3D = torch.Tensor(melspec)
data4D = data3D.unsqueeze(1)
data3D = data3D.transpose(1, 2)

print(f"input data shape: {data4D.shape}")

for model in models:
    model_maest = maest(model)
#      pipe = pipeline("audio-classification", model=f"mtg-upf/{model}")
#  
#      hf_preds = pipe(audio)
#      maest_preds, labels = model_maest.predict_labels(audio)
#      print("hf_preds", hf_preds)
#      maest_preds_pp = {labels[i]: maest_preds[i] for i in np.argsort(maest_preds)[-5:][::-1]}
#      print("maest_preds", maest_preds_pp)
#  
#  
#  print("finish!")
#  exit()

# logits_ast = model_ast(data3D).logits
    logits_maest, _ = model_maest(data4D)
    print(f"logits_maest shape: {logits_maest.shape}")

    # preds_ast = torch.mean(torch.sigmoid(logits_ast), dim=0)
    preds_maest = torch.mean(torch.sigmoid(logits_maest), dim=0)
    print(preds_maest.shape)
    np.save(f"preds_maest.npy", preds_maest.detach().numpy())

    data_essentia = data4D.numpy().astype("float32")
    data_essentia = np.swapaxes(data_essentia, 2, 3)
    print(f"input data essentia shape: {data_essentia.shape}")
    data = Pool()
    data.set('serving_default_melspectrogram', data_essentia)

    outputs = [f'StatefulPartitionedCall:{i}' for i in range(1)]
    graph_filename = "/home/palonso/exp/231016.export.onnx/output_models/discogs-maest-30s-pw-1.pb"
    model = TensorflowPredict(
        graphFilename=graph_filename,
        inputs=['serving_default_melspectrogram'],
        outputs=outputs,
    )
    preds_ess = model(data)[outputs[0]].squeeze()

    print(f"logits_ess shape: {preds_ess.shape}")

    # print("AST")
    # for i, l in enumerate(preds_ast.argsort()[-5:], 1):
    #     print('{}: {} ({:.2f}%)'.format(i, discogs_labels[l], preds_ast[l] * 100))

    print("MAEST")
    for i, l in enumerate(preds_maest.argsort()[-5:], 1):
        print('{}: {} ({:.4f}%)'.format(i, discogs_labels[l], preds_maest[l] * 100))

    print("Essentia")
    for i, l in enumerate(preds_ess.argsort()[-5:], 1):
        print('{}: {} ({:.4f}%)'.format(i, discogs_labels[l], preds_ess[l] * 100))
