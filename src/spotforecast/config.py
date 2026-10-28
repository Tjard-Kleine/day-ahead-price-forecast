import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str) -> dict:
    """Loads configs/<name>.toml."""
    with open(CONFIG_DIR / f"{name}.toml", "rb") as f:
        return tomllib.load(f)
