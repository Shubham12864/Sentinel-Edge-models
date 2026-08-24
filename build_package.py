"""Rebuild sentinel/_app cleanly from the tier folders (never leaks dev state).

Usage:  python3 build_package.py     # then: python3 -m build --wheel && twine upload
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "sentinel" / "_app"

CLEAN_CAMERAS_YAML = """\
# Tier 1 camera registry - edited automatically by Command HQ / sentinel wizard.
# source: rtsp:// URL | device index | media file path
# (fresh install: empty -- map your cameras in the console or via `sentinel setup`)
cameras: {}
"""

def rmrf(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

def copy_tree(src: Path, dst: Path, skip_names: set[str]):
    rmrf(dst)
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data", "test_*",
                                      ".pytest_cache", "*.egg-info"),
        dirs_exist_ok=True,
    )

def main():
    rmrf(APP)
    APP.mkdir(parents=True)

    # ---- tier 1: code + a CLEAN registry template (no dev paths) ----------
    t1 = APP / "tier1-ingest"
    copy_tree(ROOT / "tier1-ingest", t1, {"data"})
    (t1 / "cameras.yaml").write_text(CLEAN_CAMERAS_YAML, encoding="utf-8")

    # ---- tier 2: src/scripts/config only ----------------------------------
    t2 = APP / "tier2-ai"
    t2.mkdir()
    for sub in ("src", "scripts", "config"):
        copy_tree(ROOT / "tier2-ai" / sub, t2 / sub, set())
    req = ROOT / "tier2-ai" / "requirements.txt"
    if req.exists():
        shutil.copy(req, t2 / "requirements.txt")
    # tier-2 allowlist ships OPEN (empty section) so runtime-added cams work;
    # operators can pin their fleet here for defense-in-depth.
    (t2 / "config" / "cameras.yaml").write_text(
        "# Tier 2 allowlist. Empty/absent 'cameras:' = accept any valid camera_id.\n"
        "# Pin your fleet here for defense-in-depth, e.g.:\n"
        "# cameras:\n"
        "#   CAM_01: {location: front gate}\n",
        encoding="utf-8")

    # ---- tier 3: server + console, NO data dir (biometrics never ship) -----
    t3 = APP / "tier3-hq"
    copy_tree(ROOT / "tier3-hq", t3, {"data"})
    (t3 / "data").mkdir()                       # empty runtime dir
    (t3 / "data" / ".gitkeep").touch()

    # ---- launcher ----------------------------------------------------------
    shutil.copy(ROOT / "run_sentinel.py", APP / "run_sentinel.py")

    # ---- report ------------------------------------------------------------
    total = sum(1 for _ in APP.rglob("*") if _.is_file())
    print(f"_app assembled: {total} files")
    for banned in ("gallery.json", "identity_index.faiss", "identity_uploads", "face_crops"):
        hits = [str(p) for p in APP.rglob(banned)]
        assert not hits, f"PRIVACY LEAK in bundle: {hits}"
    assert (APP / "run_sentinel.py").exists(), "launcher missing!"
    print("privacy sweep: clean (no galleries/uploads/crops)")
    print("launcher: present")

if __name__ == "__main__":
    main()
