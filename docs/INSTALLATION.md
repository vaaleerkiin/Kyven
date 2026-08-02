# Portable Windows installation

Kyven is installed inside the repository folder. It does not require an EXE installer,
administrator access, registry changes, or a system-wide Python environment.

## Requirements

- Windows 10 or 11;
- Python 3.10-3.13 (Python 3.12 preferred);
- a compatible NVIDIA GPU and driver for practical GPU inference (CPU fallback is supported);
- internet access for the first dependency and model download;
- Foundry Nuke for the current host adapter.

## Install

1. Clone or extract Kyven into its final writable folder.
2. Double-click `install.cmd`, or run this from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

3. Choose one or more model numbers. Press Enter for the recommended 8 GB setup: SAM 2.1 Small and
   ViTMatte Small.
4. Add the path printed by the installer to the user's existing `.nuke/init.py`:

```python
import nuke

nuke.pluginAddPath("D:/Kyven/hosts/nuke")
```

Replace `D:/Kyven` with the actual repository location. Kyven discovers its root from that plugin
path; setting `KYVEN_ROOT` is normally unnecessary. Restart Nuke after editing `init.py`.

## What is created

| Path | Contents |
| --- | --- |
| `.venv/` | Private Python, PyTorch, SAM 2, ViTMatte, and ONNX Runtime |
| `models/` | Selected checkpoints verified by size and SHA-256 |
| `.runtime/pip-cache/` | Reusable installer download cache |
| `.runtime/server.token` | Private local bearer token |
| `.runtime/server.log` | Latest server output |
| `.runtime/nuke_cache/` | Per-node exported frames, raw/displayed mattes, and trimap previews |

Nothing is added to the system `PATH`, and the installer intentionally does not edit Nuke's
`init.py`.

## Update

Pull or replace the repository files, then run `install.cmd` again. Existing verified checkpoints
and downloads are reused. Restart Nuke so Python modules and API 22 are reloaded, then update older
Groups from the Nodes menu:

- `Kyven > Upgrade Selected Segment Node`;
- `Kyven > Upgrade Selected Refine Node`;
- `Kyven > Upgrade Selected Inpaint Node`.

These operations preserve node UUIDs and cached results. Upgrading Inpaint also adds the current
shared Cache controls without deleting an existing result sequence.

## Move the repository

A Windows virtual environment stores absolute paths. If the Kyven folder is moved, rerun
`install.ps1` in the new location and update `nuke.pluginAddPath(...)` in `init.py`.

## Unattended model selection

```powershell
.\install.ps1 -Model sam2.1-tiny
.\install.ps1 -Model sam2.1-small,vitmatte-small-composition-1k
.\install.ps1 -Model vitmatte-base-distinctions-646
.\install.ps1 -Model lama-2025jan-onnx
.\install.ps1 -Model big-lama-native
.\install.ps1 -Model none
```

## Install or remove models from Nuke

After the portable runtime is installed once, additional checkpoints do not require rerunning the
installer:

1. Open `Kyven > Model Manager...`, or press **Model Manager...** in a Kyven node.
2. Choose **Install Selected** and select a catalog model.
3. Confirm the progress window reaches checksum verification.
4. Press **Refresh Models** in an open node if its list was created by an older Kyven version.

Choose **Remove Selected** to reclaim disk space. Removal waits for active processing, unloads only
the selected provider, and leaves source media, rendered node caches, and other models untouched.
The manager never accepts arbitrary URLs: it uses only the pinned, commercially approved catalog.

See [Model selection](MODELS.md) for VRAM guidance and [Troubleshooting](TROUBLESHOOTING.md) if the
server or CUDA runtime does not start.
