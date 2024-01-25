from pathlib import Path
import pickle as pk


in_dir = "datasets/mtt/"


def process(file):
    return file[:-8] + ".mmap"


    print("processing", file)
    with open(file, "rb") as pfile:
        data = pk.load(pfile)

    data = {process(k): v for k, v in data.items()}

    with open(file, "wb") as pfile:
        pk.dump(data, pfile)

