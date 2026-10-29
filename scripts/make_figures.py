"""The figures of the README: the final model on the test split (Q1 2025), saved to reports/figures/.

The test split was evaluated once in step 7 (reports/test_results.md); these figures only show that result.

    python scripts/make_figures.py
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from spotforecast.models.baselines import naive_forecasts, stack_windows
from spotforecast.predict import (COLORS, CONTEXT_COLOR, FIGURE_DIR, LABELS, MUTED_COLOR, FinalModel, plot_day,
                                  style_axes, to_eur)


def main():
    model = FinalModel(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ds = model.splits["test"]
    X, Y = stack_windows(ds)
    forecasts = model.forecast(X, Y)
    actual = to_eur(Y[:, :, 0], ds)
    abs_err = {name: np.abs(to_eur(pred, ds) - actual)
               for name, pred in {"day D-1": naive_forecasts(X, Y.shape[1])["day D-1"], **forecasts}.items()}
    daily_mae = abs_err["final model"].mean(axis=1)
    days = ds.forecast_starts()
    print("test MAE in EUR/MWh, as in reports/test_results.md: "
          + ", ".join(f"{name} {err.mean():.1f}" for name, err in abs_err.items()))

    # four different days: a typical one, the price extremes and the worst forecast
    candidates = {"typical day (median error)": [np.argsort(daily_mae)[len(daily_mae) // 2]],
                  "highest price": [np.argmax(actual.max(axis=1))],
                  "lowest price": [np.argmin(actual.min(axis=1))],
                  "largest error": np.argsort(daily_mae)[::-1]}
    picks = {}
    for title, indices in candidates.items():
        picks[title] = next(i for i in indices if i not in picks.values())

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
    for ax, (title, i) in zip(axes.flat, picks.items()):
        plot_day(ax, ds, i, {name: pred[i] for name, pred in forecasts.items()},
                 f"{title}: {days[i]:%a %d %b %Y}, MAE {daily_mae[i]:.0f} EUR/MWh")
    for ax in axes[-1]:
        ax.set_xlabel("hours relative to midnight before the forecast day")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9, labelcolor=MUTED_COLOR)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "test_days.png", dpi=150, facecolor="white")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    hours = np.arange(Y.shape[1])
    ax.plot(hours, abs_err["day D-1"].mean(axis=0), color=CONTEXT_COLOR, linewidth=1.5, linestyle="--",
            label="day D-1 (baseline)")
    for name in ["ridge", "lstm mean", "final model"]:
        ax.plot(hours, abs_err[name].mean(axis=0), color=COLORS[name], linewidth=1.8, marker="o", markersize=4,
                label=LABELS[name])
    style_axes(ax, "Mean absolute error per hour of the day, test split Q1 2025")
    ax.set_xlabel("hour of the forecast day")
    ax.set_ylabel("MAE in EUR/MWh")
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED_COLOR)
    fig.savefig(FIGURE_DIR / "error_by_hour.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"figures saved to {FIGURE_DIR}")


if __name__ == '__main__':
    main()
