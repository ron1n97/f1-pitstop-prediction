"""
Download Kaggle competition data into data/.

Requires ~/.kaggle/kaggle.json with your API credentials.
Run once before training:
    python download_data.py
"""
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

COMPETITION = "playground-series-s6e5"


def main():
    print(f"Downloading {COMPETITION} → {DATA_DIR}/")
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
        capture_output=False,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)

    # Unzip
    import zipfile
    for zf in DATA_DIR.glob("*.zip"):
        print(f"Extracting {zf.name}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(DATA_DIR)
        zf.unlink()

    print("Done. Files in data/:")
    for f in sorted(DATA_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
