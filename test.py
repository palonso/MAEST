from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs

audio = MonoLoader(filename="/home/palonso/rock.mp3", sampleRate=16000, resampleQuality=4)()
model = TensorflowPredictEffnetDiscogs(graphFilename="/media/data/models/essentia-models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
embeddings = model(audio)

