"""Removes the optimizer state from the final models in models/final/, so that they are small enough for the
repository (about 6 instead of 17 MB each). The models forecast exactly the same afterwards.

    python scripts/export_final_models.py
"""
import torch

from spotforecast.final import FINAL_DIR


def main():
    for path in sorted(FINAL_DIR.glob("*.pth")):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if set(state) == {"model", "opt"}:
            # utils.load_model also reads a file that holds only the weights
            torch.save(state["model"], path)
            print(f"{path.name}: optimizer state removed, now {path.stat().st_size / 1e6:.1f} MB")
        else:
            print(f"{path.name}: has no optimizer state")


if __name__ == '__main__':
    main()
