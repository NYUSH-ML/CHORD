from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"environment variable {name} is required by configuration")

    return ENV_PATTERN.sub(replace, value)


def load_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    extends = config.pop("extends", None)
    if extends:
        base_path = Path(extends)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base = load_config(base_path)
        base.pop("_config_path", None)
        config = _deep_merge(base, config)
    config = _expand_environment(config)
    config["_config_path"] = str(config_path)
    return config


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def config_hash(config: Dict[str, Any]) -> str:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def resolve_path(config: Dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Configs in this repository use paths relative to the repository root.
    config_path = Path(config["_config_path"])
    root = config_path.parent
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    return root / path
