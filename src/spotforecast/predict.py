"""Forecasts the 24 hourly prices of a day with the final model of step 7.

The final model is the mean of the forecasts of the LSTMs in models/final/ and of Ridge, 50/50. Ridge is not stored:
it is refitted on train + validation as in spotforecast.final, which is deterministic and takes a few seconds. The
model uses the measured values of day D as perfect forecasts, so it can forecast every day of the dataset that has a
full history, but no day after the end of the data. Only the days of the test split (Q1 2025) were not seen in
training.

    python -m spotforecast.predict --date 2025-02-14           # the 24 prices as a table
    python -m spotforecast.predict --date 2025-02-14 --plot    # and the figure reports/figures/forecast_2025-02-14.png
"""
import argparse
import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset

from spotforecast.config import PROJECT_ROOT, load_config
from spotforecast.data.dataset import EnergyDataset, load_datasets
from spotforecast.evaluate import load_run, model_forecast
from spotforecast.final import FINAL_DIR
from spotforecast.models.baselines import fit_ridge, ridge_features

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
# the forecasts have the same colors in all figures (slots 1 to 3 of a categorical palette that is validated for color
# vision deficiencies); the actual price has the text color, known prices and baselines are gray context
COLORS = {"final model": "#2a78d6", "lstm mean": "#eb6834", "ridge": "#1baf7a"}
LABELS = {"final model": "final model (LSTM + Ridge)", "lstm mean": "LSTM, mean of 3 seeds", "ridge": "Ridge"}
TEXT_COLOR, MUTED_COLOR, CONTEXT_COLOR, GRID_COLOR = "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3"


class FinalModel:
    """The final model of step 7: the mean of the LSTMs in models/final/ and of Ridge, 50/50."""

    def __init__(self, device: torch.device):
        runs = sorted(p.stem for p in FINAL_DIR.glob("*.json"))
        if not runs:
            raise FileNotFoundError(f"No final models in {FINAL_DIR}, they are trained by python -m spotforecast.final")
        loaded = [load_run(run, device, FINAL_DIR) for run in runs]
        self.lstms = [model_forecast(model, device) for model, _ in loaded]
        self.config = loaded[0][1]
        with contextlib.redirect_stdout(io.StringIO()):  # the preprocessing log is long
            datasets = load_datasets(self.config, load_config("data"))
        self.splits = dict(zip(["train", "validation", "test"], datasets))
        # the same days as in spotforecast.final, and Ridge is deterministic, so it is the same model
        self.ridge = fit_ridge(ConcatDataset([self.splits["train"], self.splits["validation"]]))

    def forecast(self, X: np.ndarray, Y: np.ndarray) -> dict[str, np.ndarray]:
        """The scaled forecasts of the final model and of its two parts for stacked windows."""
        lstm = np.mean([forecast(X, Y) for forecast in self.lstms], axis=0)
        ridge = self.ridge.predict(ridge_features(X, Y))
        return {"final model": (lstm + ridge) / 2, "lstm mean": lstm, "ridge": ridge}

    def find_day(self, date: str) -> tuple[str, EnergyDataset, int]:
        """The split, the dataset and the index of the window that forecasts a day."""
        day = pd.Timestamp(date).date()
        for split, ds in self.splits.items():
            matches = np.flatnonzero(ds.forecast_starts().date == day)
            if len(matches):
                return split, ds, int(matches[0])
        raise ValueError(f"No forecast for {date}: a day needs {self.config['forecast_history'] // 24} days of history "
                         f"in the data, and the days with a clock change are skipped")


def to_eur(values: np.ndarray, ds: EnergyDataset) -> np.ndarray:
    """Scaled spot prices in EUR/MWh."""
    return values * ds.target_scaler.scale_[0] + ds.target_scaler.mean_[0]


def style_axes(ax: plt.Axes, title: str) -> None:
    """Recessive grid and axes, so that the forecasts stand out."""
    ax.set_title(title, color=TEXT_COLOR, fontsize=10, loc="left")
    ax.grid(color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=MUTED_COLOR, labelsize=8)
    ax.yaxis.label.set_color(MUTED_COLOR)
    ax.xaxis.label.set_color(MUTED_COLOR)


def plot_day(ax: plt.Axes, ds: EnergyDataset, index: int, forecasts: dict[str, np.ndarray], title: str,
             history_hours: int=48) -> None:
    """The known prices before the forecast day, its actual prices and the forecasts, in EUR/MWh."""
    x, y = ds[index]
    # the spot price in the history is scaled like the target
    ax.plot(np.arange(-history_hours, 0), to_eur(x[-history_hours:, 0].numpy(), ds), color=CONTEXT_COLOR,
            linewidth=1.5, label="known prices")
    ax.plot(np.arange(24), to_eur(y[:, 0].numpy(), ds), color=TEXT_COLOR, linewidth=2.2, label="actual price")
    for name, pred in forecasts.items():
        ax.plot(np.arange(24), to_eur(pred, ds), color=COLORS[name], linewidth=1.8, label=LABELS[name])
    ax.axvline(0, color=CONTEXT_COLOR, linewidth=0.8, linestyle=":")
    style_axes(ax, title)
    ax.set_ylabel("EUR/MWh")


def main():
    parser = argparse.ArgumentParser(description="Forecasts the 24 hourly prices of a day with the final model.")
    parser.add_argument("--date", required=True, help="the day to forecast, e.g. 2025-02-14")
    parser.add_argument("--plot", action="store_true", help="also save a figure to reports/figures/")
    args = parser.parse_args()

    model = FinalModel(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    split, ds, index = model.find_day(args.date)
    X, Y = (tensor.numpy()[None] for tensor in ds[index])
    forecasts = {name: pred[0] for name, pred in model.forecast(X, Y).items()}

    table = pd.DataFrame({"actual": to_eur(Y[0, :, 0], ds), **{name: to_eur(pred, ds) for name, pred in forecasts.items()}})
    table.index.name = "hour"
    seen = ("test split, not seen in training" if split == "test"
            else f"{split} split: the final model was trained on this day, so the forecast is optimistic")
    print(f"Forecast for {pd.Timestamp(args.date):%a %Y-%m-%d} ({seen}), EUR/MWh:\n")
    print(table.round(1).to_string())
    print("\nMAE of the day: " + ", ".join(f"{name} {np.abs(table[name] - table['actual']).mean():.1f}"
                                          for name in forecasts))

    if args.plot:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot_day(ax, ds, index, forecasts, f"Day-ahead forecast for {pd.Timestamp(args.date):%a %d %b %Y}")
        ax.set_xlabel("hours relative to midnight before the forecast day")
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED_COLOR)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        path = FIGURE_DIR / f"forecast_{pd.Timestamp(args.date):%Y-%m-%d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"figure saved to {path}")


if __name__ == '__main__':
    main()
