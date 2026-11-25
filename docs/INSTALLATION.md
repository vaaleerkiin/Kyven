# Portable Windows installation

[Documentation home](README.md) · [Nuke workflow](NUKE.md) · [Models](MODELS.md) ·
[Troubleshooting](TROUBLESHOOTING.md)

Kyven is installed inside the repository folder. It does not require an EXE installer,
administrator access, registry changes, or a system-wide Python environment.

## Requirements

- Windows 10 or 11;
- no separate Python installation is needed; a private Python 3.12 is downloaded if necessary;
- a compatible NVIDIA GPU and driver for practical GPU inference (Segment, Refine and Inpaint support CPU fallback; optional DLSS requires Windows and an NVIDIA RTX GPU);
- internet access for the first dependency and model download;
- Foundry Nuke for the current host adapter.

## What is ready after checkout

| Tool | Included | Setup before use |
| --- | --- | --- |
| WAFT SmartVector | Gizmo, callbacks and both embedded models through Git LFS | Connect `hosts/nuke` and restart Nuke 16 |
| DLSS | Worker, shim and pinned runtime metadata | Use **Model Manager > Kyven DLSS Runtime > Install Selected** |
| Segment / Refine / Inpaint | Adapters, providers and model catalog | Choose and install the model versions you need |
| Generative Fill | Native node and standalone worker code | Select the Klein bundle in Model Manager; its components install automatically |

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

### Configure Nuke and install through Model Manager

1. Clone or extract Kyven into its final writable folder. No separate Python
   installation is required.
2. Add the plugin path to your `.nuke/init.py`:

```python
import nuke

nuke.pluginAddPath("D:/Kyven/hosts/nuke")
```

3. Restart Nuke and open **Nodes > Kyven > Model Manager**. The panel opens even
   before `.venv` or the server exists.
4. Select a model and **Install Selected**. Missing base components, the model's
   runtime and its weights install in order. Setup stays asynchronous and can be
   cancelled. Runtime changes wait for active processing to finish.

**Kyven Base Runtime** installs/verifies the common environment. After setup, the
manager reopens with the full server catalog, including DLSS. **Install Required
Components** sets up only dependencies for a selected model without downloading
its weights. If no compatible Python is found, setup downloads a pinned standalone
Python 3.12, verifies its size and SHA-256, and extracts it into `.runtime/python`.
It does not register Python or modify system PATH. Existing Python 3.10-3.13
interpreters remain supported, including an explicit `-PythonExe` override.
Generative Fill installs its separate environment, Klein, Qwen and
VAE through the same manager; no separate node setup or console command is required.

Replace `D:/Kyven` with the actual repository location. Kyven discovers its root
from that plugin path; setting `KYVEN_ROOT` is normally unnecessary.

### Optional installation outside Nuke

`install.cmd` or `powershell -ExecutionPolicy Bypass -File .\install.ps1` uses the
same installers. Press Enter for SAM 2.1 Small and ViTMatte Small, or choose **12**
for Generative Fill (10.83 GB model bundle; 8 GB GPU and 32 GB RAM recommended).
The installer prints the Nuke plugin path at the end.

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
| `.runtime/python/` | Automatically downloaded private Python, verified archive and bundled licenses |
| `.venv/` | Private Python, PyTorch, SAM 2, ViTMatte, and ONNX Runtime |
| `models/` | Selected checkpoints verified by size and SHA-256 |
| `.runtime/pip-cache/` | Reusable installer download cache |
| `.runtime/server.token` | Private local bearer token |
| `.runtime/server.log` | Latest server output |
| `.runtime/nuke_cache/` | Legacy UUID-based node caches; current artist-selected caches live under Cache Folder / Name / Version |

Nothing is added to the system `PATH`, and the installer intentionally does not edit Nuke's
`init.py`.

## Update

Pull or replace the repository files and restart Nuke. In Model Manager, use
**Install Required Components** on **Kyven Base Runtime** or a selected model to
verify/install its requirements. Existing verified checkpoints and downloads are
reused. The optional `install.cmd` route remains available. Update older
Groups from the Nodes menu:

- `Kyven > Upgrade Selected Segment Node`;
- `Kyven > Upgrade Selected Refine Node`;
- `Kyven > Upgrade Selected Inpaint Node`;
- `Kyven > Upgrade Selected DLSS Node`.

These operations preserve node UUIDs and cached results. Upgrading Inpaint also adds the current
shared Cache controls without deleting an existing result sequence.

## Move the repository

A Windows virtual environment stores absolute paths. If the Kyven folder is moved,
update `nuke.pluginAddPath(...)` in `init.py`, restart Nuke and open Model Manager
to install missing components. The standalone installer can also prepare the new location.

## Unattended model selection

```powershell
.\install.ps1 -Model sam2.1-tiny
.\install.ps1 -Model sam2.1-small,vitmatte-small-composition-1k
.\install.ps1 -Model hq-sam2.1-large
.\install.ps1 -Model vitmatte-base-distinctions-646
.\install.ps1 -Model lama-2025jan-onnx
.\install.ps1 -Model big-lama-native
.\install.ps1 -Model flux2-klein-4b-q4-lanpaint
.\install.ps1 -Model none
```

The installer installs pinned PyTorch before building SAM/HQ-SAM. The optional
SAM CUDA extension is disabled on Windows, so Visual Studio and the CUDA toolkit
are not required for installation. PyTorch CUDA inference remains available.
Generative Fill resolves its dependencies in a separate environment and reuses
the core PyTorch installation without replacing core Transformers. Git is not
required for its pinned source archives. Runtime verification checks the actual
Klein/Qwen/GGUF adapter and the SciPy/OpenCV compositing dependencies.

## Install or remove models from Nuke

Model Manager can prepare the base environment even before the server exists and
installs any catalog-declared runtime required by a selected model:

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
