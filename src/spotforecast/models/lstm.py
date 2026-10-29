import torch
import torch.nn as nn
from functools import partial

from spotforecast.utils import init_weights_normal


class EnergyLSTMNet(nn.Module):
    """LSTM encoder over the history window whose last state is combined with the future covariates in an MLP head.
    The network predicts the deviation from the last known spot price for every hour of the forecast horizon.

    Expects x = (history, future) with history of shape (batch, forecast_history, n_hist_features), the spot price
    being the first history feature, and future of shape (batch, forecast_horizon, n_future_features).
    """

    def __init__(self, n_hist_features: int, n_future_features: int, forecast_horizon: int, hidden_dim: int,
                 dropout_p: float, num_layers=2, bidirectional=False, zero_init_output=False, rng: torch.Generator = None):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.input_dropout = nn.Dropout(dropout_p)

        self.lstm = nn.LSTM(
            input_size=n_hist_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        lstm_out_dim = (2 if bidirectional else 1) * hidden_dim

        self.ln = nn.LayerNorm(lstm_out_dim)

        future_flat_dim = n_future_features * forecast_horizon
        combined_dim = lstm_out_dim + future_flat_dim

        self.fc_hidden = nn.Linear(combined_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
        self.reg_layer = nn.Linear(hidden_dim, forecast_horizon)

        if rng is not None:
            self.apply(partial(init_weights_normal, generator=rng))
        if zero_init_output:
            # the untrained model then forecasts the last known price (persistence) instead of large random deviations
            nn.init.zeros_(self.reg_layer.weight)
            nn.init.zeros_(self.reg_layer.bias)

    def forward(self, x):
        hist_feats, future_feats = x

        # taken before the input dropout, otherwise the residual anchor is randomly zeroed during training
        last_known_spot = hist_feats[:, -1, 0:1]

        hist_feats = self.input_dropout(hist_feats)

        # final states of the last layer: forward after the last hour and, if bidirectional, backward after the first
        # hour. lstm_out[:, -1] would hold a backward state that has only seen the last hour
        _, (h_n, _) = self.lstm(hist_feats)
        last_out = torch.cat(list(h_n[-(2 if self.bidirectional else 1):]), dim=1)
        last_out = self.ln(last_out)

        combined_input = torch.cat([last_out, future_feats.flatten(start_dim=1)], dim=1)

        h = self.fc_hidden(combined_input)
        h = self.activation(h)
        h = self.dropout(h)
        delta_prediction = self.reg_layer(h)

        last_spot_expanded = last_known_spot.repeat(1, self.forecast_horizon)
        out = last_spot_expanded + delta_prediction

        return out


def build_model(config: dict, n_hist_features: int, n_future_features: int, rng: torch.Generator = None) -> EnergyLSTMNet:
    """Instantiates the model described by the model config for data with the given feature dimensions."""
    return EnergyLSTMNet(**config["net_params"],
                         n_hist_features=n_hist_features,
                         n_future_features=n_future_features,
                         forecast_horizon=config["forecast_horizon"],
                         rng=rng)
