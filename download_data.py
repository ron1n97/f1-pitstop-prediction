"""
Download Kaggle competition data + original source dataset into data/.

Requires ~/.kaggle/kaggle.json with your API credentials.
Run once before training:
    python download_data.py
"""
import subprocess
import sys
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ORIG_DIR = DATA_DIR / "original"
DATA_DIR.mkdir(exist_ok=True)
ORIG_DIR.mkdir(exist_ok=True)

COMPETITION = "playground-series-s6e5"
# Source dataset referenced by the competition (101k rows)
ORIGINAL_DATASET = "vanshjasuja16/f1-strategy-dataset-pit-stop-prediction"


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def _unzip_all(folder: Path) -> None:
    for zf in folder.glob("*.zip"):
        print(f"Extracting {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(folder)
        zf.unlink()


def main():
    print(f"Downloading {COMPETITION} → {DATA_DIR}/")
    _run(["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)])
    _unzip_all(DATA_DIR)

    print(f"\nDownloading original dataset → {ORIG_DIR}/")
    _run([
        "kaggle", "datasets", "download",
        "-d", ORIGINAL_DATASET,
        "-p", str(ORIG_DIR),
        "--unzip",
    ])

    # Accept either direct CSV or nested extract
    candidates = list(ORIG_DIR.rglob("f1_strategy_dataset_v4.csv"))
    if not candidates:
        print("WARNING: f1_strategy_dataset_v4.csv not found under data/original/")
        print("Place it manually at data/original/f1_strategy_dataset_v4.csv")
    elif candidates[0] != ORIG_DIR / "f1_strategy_dataset_v4.csv":
        target = ORIG_DIR / "f1_strategy_dataset_v4.csv"
        candidates[0].replace(target)
        print(f"Moved → {target}")

    print("\nDone. Files:")
    for f in sorted(DATA_DIR.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(DATA_DIR)}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
