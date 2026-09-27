"""Configuration helpers (YAML).

Loads configs with support for environment variable expansion ($VAR or ${VAR}),
cross-platform relative paths, and programmatic overrides.
"""
from __future__ import annotations

import copy
import os
import re
import yaml

DEFAULT_CONFIG = os.path.join("configs", "pipeline.yaml")
ENV_VAR_PATTERN = re.compile(r"\$(?:([A-Za-z0-9_]+)|\{([A-Za-z0-9_]+)\})")


def _expand_env_vars(obj):
    """Recursively expand environment variables in string values."""
    if isinstance(obj, str):
        def replacer(match):
            var_name = match.group(1) or match.group(2)
            return os.getenv(var_name, match.group(0))
        expanded = ENV_VAR_PATTERN.sub(replacer, obj)
        return os.path.normpath(expanded) if ("/" in expanded or "\\" in expanded) else expanded
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    """Load a YAML config; environment variables in paths are expanded,
    and `overrides` is a nested dict merged on top."""
    cfg_path = path or DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg = _expand_env_vars(cfg)

    # Apply environment variable overrides for paths if present
    if "paths" in cfg:
        paths = cfg["paths"]
        if "DATASET_DIR" in os.environ:
            paths["dataset_dir"] = os.environ["DATASET_DIR"]
        if "PROCESSED_DIR" in os.environ:
            paths["processed_dir"] = os.environ["PROCESSED_DIR"]
        if "MODELS_DIR" in os.environ:
            paths["models_dir"] = os.environ["MODELS_DIR"]
        if "OUTPUT_DIR" in os.environ:
            paths["output_dir"] = os.environ["OUTPUT_DIR"]
        if "RESULTS_DIR" in os.environ:
            paths["results_dir"] = os.environ["RESULTS_DIR"]
        if "VALIDATOR_PATH" in os.environ:
            paths["validator"] = os.environ["VALIDATOR_PATH"]

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
