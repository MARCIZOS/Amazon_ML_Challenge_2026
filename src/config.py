"""Configuration helpers (YAML)."""
from __future__ import annotations

import copy
import os

import yaml

DEFAULT_CONFIG = os.path.join("configs", "pipeline.yaml")


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML config; `overrides` is a nested dict merged on top."""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out
