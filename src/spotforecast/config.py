import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def merged(base: dict, override: dict) -> dict:
    """The base config with the values of override; tables are merged key by key, everything else is replaced."""
    result = dict(base)
    for key, value in override.items():
        result[key] = merged(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(name: str) -> dict:
    """Loads configs/<name>.toml. A config with base = "<other>" extends configs/<other>.toml."""
    with open(CONFIG_DIR / f"{name}.toml", "rb") as f:
        config = tomllib.load(f)
    base = config.pop("base", None)
    return merged(load_config(base), config) if base else config


def load_data_config(config: dict) -> dict:
    """The data config that a model config names with "data". Without it, configs/data.toml, the course data of
    phase 1, so that the runs of phase 1 load the data they were trained on."""
    return load_config(config.get("data", "data"))
