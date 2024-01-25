from librosa import load
from transformers import pipeline

from models.maest import maest

audio, sr = load("/home/palonso/rock.mp3", sr=16000)

models = [
    "discogs-maest-30s-pw-73e-ts",
    "discogs-maest-10s-fs-129e",
    "discogs-maest-10s-pw-129e",
    "discogs-maest-10s-dw-75e",
    "discogs-maest-5s-pw-129e",
    "discogs-maest-20s-pw-129e",
    "discogs-maest-30s-pw-129e",
]

for model in models:
    print(f"Model: {model}")

    time = int(model.split("-")[2].split("s")[0])
    audio = audio[:time * 16000]

    pipe = pipeline("audio-classification", model=f"mtg-upf/{model}", trust_remote_code=True)
    model = maest(arch=model)

    res = pipe(audio)
    print("hugging face")
    for i, value in enumerate(res):
        print(f"{i}: {value['label']} ({value['score'] * 100:.2f}%)")

    print("maest")
    activations, labels = model.predict_labels(audio)
    for i, l in enumerate(activations.argsort()[-5:][::-1], 1):
        print('{}: {} ({:.2f}%)'.format(i, labels[l], activations[l] * 100))

   from essentia.standard import TensorflowPredict
   from essentia import Pool
   import numpy as np



   data = Pool()
   data.set('serving_default_input_values', tensor)


   outputs = [f'StatefulPartitionedCall:{i}' for i in range(14)]
   graph_filename = "output_models/discogs-maest-30s-pw-73e-ts.pb"
   model = TensorflowPredict(
       graphFilename=graph_filename,
       inputs=['serving_default_input_values'],
       outputs=outputs,
   )

   data_out = model(data)

   for output in outputs:
       print(f"printing {output} shape")
       print(data_out[output].shape)
