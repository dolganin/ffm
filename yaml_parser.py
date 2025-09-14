from pathlib import Path
import yaml


def load_config(path):
    """Load a YAML configuration file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class YamlParser:
    """Simple wrapper object matching the interface expected by ``train.py``."""

    def __init__(self, path: str):
        self.path = path

    def get_config(self):
        return load_config(self.path)
