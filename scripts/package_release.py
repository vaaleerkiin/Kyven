"""Build an installable Kyven distribution with DLSS and embedded WAFT models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

FOLDERS = ("src", "hosts", "native", "docs", "requirements", "scripts", "plugins")
ROOT_FILES = ("install.cmd", "install.ps1", "pyproject.toml", "README.md", "LICENSE",
              "THIRD_PARTY_NOTICES.md", "PROJECT_BIBLE.md", "CODEX.md", "OWNER_NOTES.md")


def package_release(root: Path, output: Path) -> Path:
    root = root.resolve()
    waft = root / "plugins/Cattery/WAFT"
    if waft.is_dir():
        model_manifest = json.loads((waft / "model-manifest.json").read_text(encoding="utf-8"))
        for name, expected in model_manifest["models"].items():
            model = waft / name
            if not model.is_file() or model.stat().st_size != expected["size_bytes"]:
                raise ValueError(f"WAFT model missing or an LFS pointer: run git lfs pull ({name})")
            with model.open("rb") as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"WAFT model checksum mismatch: {name}")
    package = root / "src/kyven/resources/dlss/windows_x64"
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    vendor = package / "nvngx_dlssnr.dll"
    # The optional local NR component is deliberately absent from a source clone.
    # Its pinned download identity remains in vendor-source.json, not a phantom ZIP member.
    if not vendor.is_file():
        manifest["sha256"].pop(vendor.name, None)
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((package / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Build the verified Kyven worker package first: {name}")
    if vendor.is_file():
        spec = json.loads((package.parent / "vendor-source.json").read_text(encoding="utf-8"))
        if (vendor.stat().st_size != spec["member_size_bytes"]
                or manifest["sha256"].get(vendor.name) != spec["member_sha256"]):
            raise ValueError("The bundled NVIDIA component must match the pinned size and manifest hash")
    files = [root / name for name in ROOT_FILES]
    for folder in FOLDERS:
        files.extend(path for path in (root / folder).rglob("*") if path.is_file()
                     and not path.is_symlink()
                     and not any(part in {"__pycache__", ".pytest_cache", ".ruff_cache"}
                                 for part in path.parts)
                     and path.suffix not in {".pyc", ".pyo", ".pdb", ".lib", ".exp"})
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".kyven-release-", suffix=".zip", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(set(files)):
                resolved = path.resolve()
                if (not resolved.is_relative_to(root) or (
                        path.name.lower() == "nvngx_dlssnr.dll" and resolved != vendor.resolve())):
                    raise ValueError(f"Refusing to bundle data outside the managed package: {path.name}")
                member = Path("Kyven") / path.relative_to(root)
                if path == package / "manifest.json":
                    archive.writestr(str(member).replace("\\", "/"), json.dumps(manifest, indent=2))
                else:
                    archive.write(path, member)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(package_release(Path(__file__).resolve().parents[1], args.output.resolve()))


if __name__ == "__main__":
    main()
