"""Configuration loading.

``configs/default.yaml`` holds every setting. Any other YAML file only needs the
keys it wants to change; it is deep-merged on top of the defaults.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


# top-level sections that an override REPLACES instead of merging: listing
# `models:` in your YAML means "exactly these models", not "these plus the defaults".
REPLACE_SECTIONS = {"models"}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any], _top: bool = True) -> Dict[str, Any]:
    """Return a new dict: ``override`` merged recursively into ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if _top and key in REPLACE_SECTIONS:
            out[key] = copy.deepcopy(value)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value, _top=False)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load default config, merge an optional YAML file and optional dict overrides."""
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG_PATH:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    _derive(cfg)
    return cfg


def _derive(cfg: Dict[str, Any]) -> None:
    """Fill values that depend on other values."""
    if cfg["validation"].get("purge_days") is None:
        cfg["validation"]["purge_days"] = int(cfg["labels"]["horizon"])


def resolve_path(p: Union[str, Path]) -> Path:
    """Relative paths in the config are relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
