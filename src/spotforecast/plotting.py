import torch
import pandas as pd
import plotly.express as px
from typing import Callable

from spotforecast.data.dataset import default_batch_transform


def predict_one_sample(model, features, targets, device, transform_batch_fn:Callable=default_batch_transform):
    x, y = transform_batch_fn(features.unsqueeze(0), targets.unsqueeze(0), device, train=False)
    if isinstance(x, tuple):
        x = tuple(_x if _x.shape[0] == 1 else _x.unsqueeze(0) for _x in x)
    pred = model(x)
    return pred, y


def plot_one_prediction(pred, y, ):
    df = pd.DataFrame(data={"pred": pred.cpu().detach().squeeze(0).numpy(), "ground truth": y.cpu().detach().squeeze(0).numpy()}, index=pd.Series(torch.arange(start=1, end=y.shape[1]+1, device="cpu"), name="hours"))
    fig = px.line(df)
    return fig
