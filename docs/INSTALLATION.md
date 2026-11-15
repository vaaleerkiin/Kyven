# Portable Windows installation

[Documentation home](README.md) · [Nuke workflow](NUKE.md) · [Models](MODELS.md) ·
[Troubleshooting](TROUBLESHOOTING.md)

Kyven is installed inside the repository folder. It does not require an EXE installer,
administrator access, registry changes, or a system-wide Python environment.

## Requirements

- Windows 10 or 11;
- Python 3.10-3.13 (Python 3.12 preferred);
- a compatible NVIDIA GPU and driver for practical GPU inference (Segment, Refine and Inpaint support CPU fallback; optional DLSS requires Windows and an NVIDIA RTX GPU);
- internet access for the first dependency and model download;
- Foundry Nuke for the current host adapter.

## What is ready after checkout

| Tool | Included | Setup before use |
| --- | --- | --- |
| WAFT SmartVector | Gizmo, callbacks and both embedded models through Git LFS | Connect `hosts/nuke` and restart Nuke 16 |
| DLSS | Worker, shim and pinned runtime metadata | Use **Model Manager > Kyven DLSS Runtime > Install Selected** |
| Segment / Refine / Inpaint | Adapters, providers and model catalog | Choose and install the model versions you need |

For WAFT alone, skip the server environment installer and connect the Nuke plugin
path described below. DLSS uses Windows/RTX and the server environment; its NVIDIA
component is not stored in Git. Full preview bundles can include it, while plain
clones download the pinned component during managed runtime installation.

## Install

### Download ZIP

Choose **Code > Download ZIP** on GitHub and extract it into a permanent writable
folder. **Include Git LFS objects in archives** is enabled for this repository,
so both WAFT `.cat` models are included as actual binaries. End users do not need
Git or Git LFS for this route.

### Clone with Git

Install Git LFS, then run:

```bash
git lfs install
git clone https://github.com/vaaleerkiin/Kyven.git
```

Both WAFT models download during cloning. For an existing clone:

```bash
cd Kyven
git lfs install
git lfs pull
```

If `GIT_LFS_SKIP_SMUDGE` was enabled, run `git lfs pull` to fetch the model
binaries. A `.cat` file should be about 124 or 244 MiB, not a small text pointer.

### Configure Nuke and server tools

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

## Standalone WAFT SmartVector

The complete local/release package includes `plugins/Cattery/WAFT/` with both
`WAFT.cat` and `WAFT_FP32.cat`. The same `hosts/nuke` plugin path above registers
WAFT; restart Nuke 16 and choose **Kyven > WAFT SmartVector**. It uses native Nuke
Inference without Kyven Server, a separate Python environment or a Model Manager download.

Both deployed models are versioned with Git LFS. Install Git LFS before cloning
to receive them automatically. For an existing checkout, run `git lfs install`
and `git lfs pull` in the repository root. A ZIP must include the actual `.cat`
files rather than LFS pointer text. The standard Kyven installer and Model Manager
do not manage these models; see [WAFT](WAFT.md) for checks and recovery.

For standalone Cattery installation, copy the complete `plugins/Cattery/WAFT/`
folder into `.nuke/Cattery/`, preserving `cat.json` directly inside `WAFT/`.
For deployment, ship the complete plugin folder with its models and license;
`development/` is unnecessary. See [WAFT setup and workflow](WAFT.md).

## What is created

| Path | Contents |
| --- | --- |
| `.venv/` | Private Python, PyTorch, SAM 2, ViTMatte, and ONNX Runtime |
| `models/` | Selected checkpoints verified by size and SHA-256 |
| `.runtime/pip-cache/` | Reusable installer download cache |
| `.runtime/server.token` | Private local bearer token |
| `.runtime/server.log` | Latest server output |
| `.runtime/nuke_cache/` | Legacy UUID-based node caches; current artist-selected caches live under Cache Folder / Name / Version |

Nothing is added to the system `PATH`, and the installer intentionally does not edit Nuke's
`init.py`.

## Update

Pull or replace the repository files, then run `install.cmd` again. Existing verified checkpoints
and downloads are reused. Restart Nuke so Python modules and API 36 are reloaded, then update older
Groups from the Nodes menu:

- `Kyven > Upgrade Selected Segment Node`;
- `Kyven > Upgrade Selected Refine Node`;
- `Kyven > Upgrade Selected Inpaint Node`;
- `Kyven > Upgrade Selected DLSS Node`.

These operations preserve node UUIDs and cached results. Upgrading Inpaint also adds the current
shared Cache controls without deleting an existing result sequence.

## Move the repository

A Windows virtual environment stores absolute paths. If the Kyven folder is moved, rerun
`install.ps1` in the new location and update `nuke.pluginAddPath(...)` in `init.py`.

## Unattended model selection

```powershell
.\install.ps1 -Model sam2.1-tiny
.\install.ps1 -Model sam2.1-small,vitmatte-small-composition-1k
.\install.ps1 -Model hq-sam2.1-large
.\install.ps1 -Model vitmatte-base-distinctions-646
.\install.ps1 -Model lama-2025jan-onnx
.\install.ps1 -Model big-lama-native
.\install.ps1 -Model none
```

## Install or remove models from Nuke

After the initial portable setup, models do not require rerunning the installer. Model Manager also
installs any catalog-declared runtime required by the selected model:

1. Open `Kyven > Model Manager...`, or press **Model Manager...** in a Kyven node.
2. Choose **Install Selected** and select a catalog model.
3. Keep the progress window open while any required runtime is installed, Kyven Server restarts,
   and the checkpoint reaches checksum verification.
4. Press **Refresh Models** in an open node if its list was created by an older Kyven version.

Choose **Remove Selected** to reclaim disk space. Removal waits for active processing, unloads only
the selected provider, and leaves source media, rendered node caches, and other models untouched.
The manager never accepts arbitrary URLs: model files and runtimes use only the pinned,
commercially approved catalog.

For optional RTX reconstruction, install **Kyven DLSS Runtime** from Model Manager and follow the [DLSS setup guide](DLSS.md#setup).

See [Model selection](MODELS.md) for VRAM guidance and [Troubleshooting](TROUBLESHOOTING.md) if the
server or CUDA runtime does not start.
