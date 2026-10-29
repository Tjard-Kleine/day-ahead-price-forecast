"""Tests that guard the pipeline against leakage: a forecast may only use what is known at 12:00 on the day before.

The tests at the end need the dataset in data/raw/ and are skipped without it.

    python -m pytest tests
"""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest

from spotforecast.config import PROJECT_ROOT, load_config
from spotforecast.data.dataset import EnergyDataset, fit_trimmed_scaler, load_datasets
from spotforecast.data.features import add_columns_, get_calendar_features, source_columns, with_derived_features
from spotforecast.data.smard import MEASURED, NEIGHBOUR_PRICES
from spotforecast.evaluate import diebold_mariano


def test_window_history_ends_right_before_its_targets():
    rows = np.arange(100, dtype=np.float32)[:, None]
    ds = EnergyDataset(rows, rows, window=10, horizon=5, target_starts=[10, 30, 97])
    assert len(ds) == 2  # the targets starting at row 97 do not fit into the data
    for i in range(len(ds)):
        x, y = ds[i]
        assert x[-1, 0] + 1 == y[0, 0]
        assert len(x) == 10 and len(y) == 5


def test_scaler_uses_only_the_training_rows():
    data = pd.DataFrame({"a": np.random.default_rng(1).normal(50, 10, 300)})
    train_map = pd.Series(np.arange(300) < 200)
    scaler = fit_trimmed_scaler(data, ["a"], train_map)
    data.loc[200:, "a"] = 1e6  # extreme values after the training rows must not change the scaling
    assert np.allclose(fit_trimmed_scaler(data, ["a"], train_map).scale_, scaler.scale_)
    assert np.allclose(fit_trimmed_scaler(data, ["a"], train_map).mean_, scaler.mean_)


def test_calendar_columns_are_not_scaled():
    data = pd.DataFrame({"saturday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0] * 10})
    scaler = fit_trimmed_scaler(data, ["saturday"], pd.Series([True] * 70))
    assert scaler.mean_[0] == 0 and scaler.scale_[0] == 1


def test_lag_columns_hold_earlier_values():
    data = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=48, freq="h"), "natural gas": np.arange(48.0)})
    add_columns_(data, ["natural gas_lag12"], "Date")
    assert (data["natural gas_lag12"].iloc[12:].to_numpy() == np.arange(36.0)).all()
    assert (data["natural gas_lag12"].iloc[:12] == 0).all()  # no earlier value, the first one is repeated


def test_calendar_features():
    dates = pd.Series(pd.to_datetime(["2024-10-03 12:00", "2024-11-01 12:00", "2024-11-16 12:00", "2024-11-17 12:00",
                                      "2024-11-19 12:00", "2024-12-24 12:00"]).tz_localize("Europe/Berlin"))
    calendar = get_calendar_features(dates)
    assert calendar["holiday"].round(2).tolist() == [1.0, 0.57, 0.0, 0.0, 0.0, 1.0]  # All Saints' Day: 5 of 16 states
    assert calendar["saturday"].tolist() == [0, 0, 1, 0, 0, 0]
    assert calendar["sunday"].tolist() == [0, 0, 0, 1, 0, 0]


def test_derived_features_need_their_sources():
    assert with_derived_features(["Spot", "solar", "solar capacity"])[-2:] == ["Spot_roll_168", "solar_util"]
    # without the measured solar generation, its utilization would leak it into a realistic model
    assert with_derived_features(["Spot", "forecast solar"]) == ["Spot", "forecast solar", "Spot_roll_168"]


@pytest.mark.parametrize("name", ["lstm_v2", "lgbm_v2"])
def test_v2_uses_only_what_is_known_at_noon_before_the_day(name):
    config = load_config(name)
    measured = set(MEASURED) | {"residual load"}  # computed from measured values
    assert not measured & set(config["features"]), "measured values must be shifted, e.g. <column>_lag12"
    assert not (measured | set(NEIGHBOUR_PRICES)) & set(config["targets"][1:]), "no inputs of day D from the auction"


def test_source_columns_of_derived_features():
    assert source_columns(["Spot", "residual load", "natural gas_lag12", "saturday"]) == \
           ["Spot", "conso", "onshore wind", "offshore wind", "solar", "natural gas"]


def test_diebold_mariano_prefers_the_smaller_errors():
    rng = np.random.default_rng(1)
    y = rng.normal(size=(90, 24))
    good, bad = y + rng.normal(scale=0.5, size=y.shape), y + rng.normal(scale=1.0, size=y.shape)
    statistic, p_value = diebold_mariano(good, bad, y)
    assert statistic < 0 and p_value < 0.01


needs_data = pytest.mark.skipif(not (PROJECT_ROOT / load_config("data")["path"]).exists(),
                                reason="the dataset is not in data/raw/")


@pytest.fixture(scope="module")
def splits():
    with contextlib.redirect_stdout(io.StringIO()):
        return dict(zip(["train", "validation", "test"], load_datasets(load_config("lstm"), load_config("data"))))


@needs_data
def test_every_window_ends_before_its_forecast_day(splits):
    for ds in splits.values():
        history_end = ds.dates[ds.usable_idxs + ds.window - 1]
        starts = ds.forecast_starts()
        target_end = ds.dates[ds.usable_idxs + ds.window + ds.horizon - 1]
        assert (starts.hour == 0).all() and (history_end.hour == 23).all()
        assert ((starts - history_end) == pd.Timedelta("1h")).all()
        assert (target_end.date == starts.date).all()  # the 24 targets are the hours of one day


@needs_data
def test_splits_do_not_overlap(splits):
    preprocessing = load_config("data")["preprocessing"]
    valid_cutoff = pd.Timestamp(preprocessing["valid_cutoff_datetime"]).tz_localize("Europe/Berlin")
    test_cutoff = pd.Timestamp(preprocessing["test_cutoff_datetime"]).tz_localize("Europe/Berlin")
    last_target = lambda ds: ds.dates[ds.usable_idxs + ds.window + ds.horizon - 1]
    assert last_target(splits["train"]).max() < valid_cutoff <= splits["validation"].forecast_starts().min()
    assert last_target(splits["validation"]).max() < test_cutoff <= splits["test"].forecast_starts().min()


@needs_data
def test_excluded_days_are_no_training_targets():
    config = dict(load_config("lstm"), exclude_days=["2024-06-26"])
    with contextlib.redirect_stdout(io.StringIO()):
        train_ds, _, _ = load_datasets(config, load_config("data"))
    days = set(train_ds.forecast_starts().strftime("%Y-%m-%d"))
    assert "2024-06-26" not in days and {"2024-06-25", "2024-06-27"} <= days


@needs_data
def test_spot_in_the_history_equals_the_target(splits):
    for ds in splits.values():
        assert (ds.features[:, 0] - ds.target[:, 0]).abs().max() < 1e-5
