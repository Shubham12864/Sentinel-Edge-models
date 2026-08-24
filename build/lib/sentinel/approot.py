"""Locate (and on first run, unpack) the bundled tier applications.

Wheel/sdist installs ship the three tiers inside sentinel/_app/.  Because the
apps write runtime data (camera registry, galleries, crops) next to their code,
the first `sentinel` command copies them to a user-writable app root:

    Windows : %LOCALAPPDATA%/sentinel-edge
    Linux   : ~/.local/share/sentinel-edge
    macOS   : ~/Library/Application Support/sentinel-edge

Repo checkouts keep using the repo directory itself.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
BUNDLED = PKG_DIR / "_app"
MARKER = ".sentinel-root"


def _user_root() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
    return base / "sentinel-edge"


def app_root() -> Path:
    """Directory containing tier1-ingest/ tier2-ai/ tier3-hq/, bootstrapping if needed."""
    repo_root = PKG_DIR.parent
    if (repo_root / "run_sentinel.py").exists():          # repo checkout
        return repo_root
    if (BUNDLED.parent / MARKER).exists() is False and BUNDLED.exists():
        pass  # wheel install path handled below

    root = _user_root()
    if not (root / "tier1-ingest").exists():              # first run: unpack
        if not BUNDLED.exists():
            raise RuntimeError(
                "Sentinel Edge application files not found. "
                "Reinstall with: pip install --force-reinstall sentinel-edge"
            )
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(BUNDLED, root, dirs_exist_ok=True)
        (root / "data").mkdir(exist_ok=True)
    return root


def tier(name: str) -> Path:
    """Path helper: tier('tier2-ai') etc."""
    p = app_root() / name
    if not p.exists():
        raise RuntimeError(f"missing application folder: {p}")
    return p
