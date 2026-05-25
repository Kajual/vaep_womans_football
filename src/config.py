"""Configuration loader.

All pipeline modules call ``load_config()`` which returns a dict-like ``Config``
with attribute access. Paths in the YAML are interpreted relative to the
repository root (discovered by walking up from the YAML file).

Override the config file by setting the environment variable ``VAEP_CONFIG``
or passing ``--config path/to/other.yaml`` on the command line of any module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Path discovery
# ---------------------------------------------------------------------------

def find_repo_root(start: Path | None = None) -> Path:
    """Return the directory containing ``pyproject.toml`` by walking upwards."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(
        "Could not find repo root (no pyproject.toml). "
        "Run from inside the project, or set VAEP_REPO_ROOT."
    )


_env_root = os.environ.get("VAEP_REPO_ROOT", "").strip()
REPO_ROOT = Path(_env_root).resolve() if _env_root else find_repo_root()


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Thin wrapper around the parsed YAML.

    Access keys as attributes (``cfg.paths.statsbomb_root``) or with ``[]``.
    """

    _raw: dict[str, Any]
    _root: Path

    # Resolved path attributes are added at construction.
    paths: "AttrDict" = field(init=False)
    competitions: "AttrDict" = field(init=False)
    split: "AttrDict" = field(init=False)
    labels: "AttrDict" = field(init=False)
    features_baseline: "AttrDict" = field(init=False)
    phases: "AttrDict" = field(init=False)
    space_features: "AttrDict" = field(init=False)
    modelling: "AttrDict" = field(init=False)
    transfer_learning: "AttrDict" = field(init=False)

    def __post_init__(self) -> None:
        for key, value in self._raw.items():
            setattr(self, key, _wrap(value))
        # Resolve every path under self.paths to an absolute Path object.
        resolved = {
            k: (self._root / v).resolve() for k, v in self._raw["paths"].items()
        }
        self.paths = AttrDict(resolved)

    @property
    def repo_root(self) -> Path:
        return self._root

    def ensure_dirs(self, *keys: str) -> None:
        """Create the named path directories if they don't exist."""
        for k in keys:
            p = getattr(self.paths, k)
            p.mkdir(parents=True, exist_ok=True)


class AttrDict(dict):
    """A dict that also supports attribute access for keys."""

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    """Recursively wrap dicts in AttrDict so config.foo.bar works."""
    if isinstance(obj, dict):
        return AttrDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None) -> Config:
    """Load and return the pipeline configuration.

    Resolution order:
      1. ``path`` argument if provided
      2. ``$VAEP_CONFIG`` environment variable
      3. ``<repo_root>/configs/default.yaml``
    """
    if path is not None:
        cfg_path = Path(path)
    elif "VAEP_CONFIG" in os.environ:
        cfg_path = Path(os.environ["VAEP_CONFIG"])
    else:
        cfg_path = REPO_ROOT / "configs" / "default.yaml"

    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()

    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(_raw=raw, _root=REPO_ROOT)


if __name__ == "__main__":  # pragma: no cover
    cfg = load_config()
    print(f"Repo root:      {cfg.repo_root}")
    print(f"StatsBomb root: {cfg.paths.statsbomb_root}")
    print(f"Inventory dir:  {cfg.paths.inventory_dir}")
    print(f"k_main:         {cfg.labels.k_main}")
