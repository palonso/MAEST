from safetensors.torch import load_file
from glob import glob
import torch
from tqdm import tqdm


def main(base_path: str):
    """
    Convert safetensors files to pytorch checkpoints files.
    Args:
        base_path (str): The base path where the safetensors files are located.
    Returns:
        None
    """
    for filename in tqdm(glob(f"{base_path}/*.safetensors")):
        ckpt = load_file(filename)
        torch.save(ckpt, filename.replace(".safetensors", ".bin"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, required=True)
    args = parser.parse_args()
    main(args.base_path)
