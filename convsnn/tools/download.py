"""Download the original dataset into the repository's ignored data directory."""
from pathlib import Path
import shutil
import kagglehub

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[2] / "data"
    if destination.exists():
        raise SystemExit("data already exists; choose or prepare the dataset manually")
    source = Path(kagglehub.dataset_download("mohitsingh1804/plantvillage"))
    shutil.copytree(source, destination)
    print(f"Dataset copied to {destination}; arrange train/ and val/ at that root before training.")
