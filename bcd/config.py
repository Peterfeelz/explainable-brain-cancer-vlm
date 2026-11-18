"""Load YAML config into a nested attribute dictionary."""
from __future__ import annotations
import copy, yaml
from pathlib import Path


class Cfg(dict):
    """dict with attribute access (cfg.gate.alpha)."""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return v
    def __setattr__(self, k, v):
        self[k] = v
    def copy(self):
        return wrap(copy.deepcopy(dict(self)))


def wrap(d):
    if isinstance(d, dict):
        return Cfg({k: wrap(v) for k, v in d.items()})
    if isinstance(d, list):
        return [wrap(v) for v in d]
    return d


def load_config(path: str | Path | None = None, **overrides) -> Cfg:
    if path is None:
        path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(path) as f:
        cfg = wrap(yaml.safe_load(f))
    for dotted, val in overrides.items():
        node = cfg
        keys = dotted.split("__")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = val
    return cfg
