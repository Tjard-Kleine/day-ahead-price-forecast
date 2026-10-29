import numpy as np
import torch
import random
from warnings import warn
from pathlib import Path
from torch.optim import Optimizer
import torch.nn as nn


def seed_torch(seed, reproducible:bool = False):
    np.random.seed(seed % 2**32)  # numpy global rng, only accepts 32 bit seeds
    torch.manual_seed(seed % (2**64-1))  # torch global rng (CPU & CUDA)
    random.seed(seed % (2**64-1))  # python global rng (MT)

    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_weights_normal(m: nn.Module, generator: torch.Generator):
    if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight, generator=generator)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def load_model(model_path: Path, model: nn.Module, optimizer: Optimizer, device: torch.device, with_opt:bool = True, strict: bool = True):
    state = torch.load(model_path, map_location=device, weights_only=True)
    hasopt = set(state)=={'model', 'opt'}
    model_state = state['model'] if hasopt else state
    model.load_state_dict(model_state, strict=strict)
    if hasopt and with_opt:
        try: optimizer.load_state_dict(state['opt'])
        except:
            if with_opt: warn("Could not load the optimizer state.")
    elif with_opt: warn("Saved file doesn't contain an optimizer state.")
