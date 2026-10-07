"""Structural tests for the MV3 browser extension (no browser required)."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / "extension"


def test_manifest_valid_mv3():
    manifest = json.loads((EXT / "manifest.json").read_text())
    assert manifest["manifest_version"] == 3
    assert manifest["name"]
    assert manifest["version"]
    assert manifest["background"]["service_worker"] == "background.js"
    assert manifest["content_scripts"][0]["js"] == ["content.js"]
    assert "storage" in manifest["permissions"]
    assert any(h.startswith("http://127.0.0.1") for h in manifest["host_permissions"])


def test_referenced_files_exist():
    manifest = json.loads((EXT / "manifest.json").read_text())
    files = [manifest["background"]["service_worker"]]
    files += manifest["content_scripts"][0]["js"]
    files += manifest["content_scripts"][0]["css"]
    files += list(manifest["action"]["default_icon"].values())
    files += list(manifest["icons"].values())
    files += ["popup.html", "popup.js"]
    missing = [f for f in files if not (EXT / f).exists()]
    assert not missing, f"missing extension files: {missing}"


def test_popup_references_assets():
    html = (EXT / "popup.html").read_text()
    assert "popup.js" in html
    assert (EXT / "popup.js").exists()


def test_js_syntax():
    node = shutil.which("node")
    if node is None:
        return  # node not available in this environment
    for name in ("background.js", "content.js", "popup.js"):
        proc = subprocess.run([node, "--check", str(EXT / name)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, f"{name}: {proc.stderr}"


def test_demo_page_exists():
    page = EXT / "demo" / "chat.html"
    assert page.exists()
    text = page.read_text()
    assert "composer" in text and "Send" in text
