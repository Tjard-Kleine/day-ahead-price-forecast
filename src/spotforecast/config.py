import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    """Loads configs/<name>.toml."""
    with open(CONFIG_DIR / f"{name}.toml", "rb") as f:
        return tomllib.load(f)


def load_data_config(config: dict) -> dict:
    """The data config that a model config names with "data". Without it, configs/data.toml, the course data of
    phase 1, so that the runs of phase 1 load the data they were trained on."""
    return load_config(config.get("data", "data"))
