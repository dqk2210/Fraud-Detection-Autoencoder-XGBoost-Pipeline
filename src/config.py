"""
config.py -- Load YAML configuration files.

Provides a single source of truth by reading from config/*.yaml.
All src/ modules should use these functions instead of hardcoding paths or params.
"""

import os
import yaml

# Project root = parent of src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")


def load_yaml(name: str) -> dict:
    """
    Load a YAML config file from the config/ directory.

    Parameters
    ----------
    name : str
        Filename (with or without .yaml extension).

    Returns
    -------
    dict : Parsed YAML content.
    """
    if not name.endswith(".yaml"):
        name += ".yaml"
    path = os.path.join(CONFIG_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths() -> dict:
    """Load paths.yaml and resolve all paths relative to PROJECT_ROOT."""
    raw = load_yaml("paths")
    resolved = {}
    for section, items in raw.items():
        resolved[section] = {}
        for key, value in items.items():
            resolved[section][key] = os.path.join(PROJECT_ROOT, value)
    return resolved


def get_model_config() -> dict:
    """Load model_config.yaml (hyperparameters per model)."""
    return load_yaml("model_config")


def get_train_config() -> dict:
    """Load train_config.yaml (training settings)."""
    return load_yaml("train_config")


# Convenience: ensure output dirs exist when config is first loaded
def ensure_output_dirs():
    """Create all output directories defined in paths.yaml if they don't exist."""
    paths = get_paths()
    for section in paths.values():
        for key, path in section.items():
            if key.endswith("_dir"):
                os.makedirs(path, exist_ok=True)
