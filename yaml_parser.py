from pathlib import Path
import yaml


def load_config(path):
    """Load a YAML configuration file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
