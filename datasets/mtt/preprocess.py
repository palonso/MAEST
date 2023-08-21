import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from glob import glob
from pathlib import Path
from tempfile import TemporaryDirectory
from shutil import copyfileobj
from subprocess import run
from urllib.request import urlopen, Request
from zipfile import ZipFile

from tqdm import tqdm


MTT_URL_BASE = "https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.00"
script = Path(__file__).parent / ".." / ".." / "helpers" / "melspectrogram_extractor.py"


def download_mtt(download_dir: Path) -> None:
    """Download the MagnaTagATune dataset."""

    for i in range(1, 4):
        request = Request(MTT_URL_BASE + str(i))
        response = urlopen(request)
        data = response.read()
        with open(download_dir / f"mp3.zip.00{i}", "wb") as f:
            f.write(data)


def extract_mtt(download_dir: Path, audio_dir: Path) -> None:
    """Extract the audios from the MagnaTagATune dataset zips."""

    with TemporaryDirectory() as tmpdir:
        with open(Path(tmpdir, 'output_file.zip'), 'w+b') as wfd:
            # Search for all files matching searchstring
            for f in glob(str(download_dir / 'mp3.zip.*')):
                with open(f, 'rb') as fd:
                    copyfileobj(fd, wfd)  # Concatenate

            with ZipFile(wfd, 'r') as zip_ref:
                zip_ref.extractall(audio_dir)


def extract_essentia_melspecs(audio_dir: Path, melspec_dir: Path, max_workers: int = 16, force:
                              bool = False) -> None:
    """Extract the mel-spectrograms from the MagnaTagATune audio."""

    mp3_files = glob(str(audio_dir / "**/*.mp3"), recursive=True)

    args = []
    for mp3_file in mp3_files:
        mp3_file = Path(mp3_file)

        melspec_file = str((melspec_dir /
                           mp3_file.relative_to(audio_dir).with_suffix(".mmap")).resolve())
        mp3_file = str(mp3_file.resolve())

        if not force and Path(melspec_file).exists():
            continue

        args.append((sys.executable, script, mp3_file, melspec_file))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tqdm(executor.map(run, args), total=len(args))


def main(args):
    """Dowload, uncompress and extract mel-spectrograms from the MagnaTagATune dataset."""

    # Make sure the target directories exist
    args.download_dir.mkdir(parents=True, exist_ok=True)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.melspec_dir.mkdir(parents=True, exist_ok=True)

    download_mtt(args.download_dir)
    extract_mtt(args.download_dir, args.audio_dir)
    extract_essentia_melspecs(args.audio_dir, args.melspec_dir, max_workers=args.num_workers,
                              force=args.force)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--download_dir", type=Path, default="data/mtt/download")
    parser.add_argument("--audio_dir", type=Path, default="data/mtt/audio")
    parser.add_argument("--melspec_dir", type=Path, default="data/mtt/melspec")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    main(args)
