# Developing WAFT from scratch

This directory contains development sources. To use the finished node, use
`plugins/Cattery/WAFT/` in the Kyven root; see the [WAFT guide](../../docs/WAFT.md).

## Retained files

- `WAFT-source/`: upstream source and bundled dependencies, including local changes.
- `RAFT/`: Cattery template gizmo, callbacks, metadata and icons. RAFT.cat is not needed to build WAFT.
- `tools/`: checkpoint download, export, packaging, diagnostics and verification scripts.
- `WAFT/`: the standalone package shell, without duplicate model binaries.
- `reports/`: logs and reports from the previous build; they do not validate a new build.
- `provenance.json`: source URL, commit, submodule state and checkpoint SHA-256.
- `upstream-local.patch`: local changes against the recorded commit, already applied in the source tree.
- `requirements-lock.txt`: installed package versions from the original Python 3.10.20 environment.

The nested Git history was removed; the source commit is recorded in provenance.

`tools/package_waft.py` also reproduces the Kyven layout and copies the shared
logo assets into the portable plugin folder. Keep `kyven_logo.png` beside the
callback module so standalone Cattery copies retain their branding.
The dinov3 submodule was not initialized in the original environment and is not
required for the WAFT-a1 export. Training datasets and the user's test video are
not included.

## 1. Create the environment

Requirements: Windows, Python 3.10 with Python Launcher (`py`), and licensed
NukeX 16 for `.cat` creation. FP16 export and CUDA tests require an NVIDIA GPU.
Open PowerShell in `development/waft`:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r tools\requirements-export.txt
```

The requirements file specifies the tested main dependencies and CUDA 11.8 index.
To repeat the complete installed package set, replace the second command with:

```powershell
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu118 -r requirements-lock.txt
```

## 2. Restore development copies of the models

Both deployed `.cat` models are stored in `../../plugins/Cattery/WAFT/` through
Git LFS. Install Git LFS before cloning; for missing models run `git lfs install`
and `git lfs pull` from the Kyven root. Then run:

```powershell
.\.venv\Scripts\python.exe tools\prepare_workspace.py
```

This copies missing package files, including both models, into
`WAFT/Cattery/WAFT/` (about 369 MiB). Existing development files are not overwritten.
Alternatively, build the models in the next step. Ordinary plugin use does not
require a development copy or this Python environment.

## 3. Rebuild from source

```powershell
.\.venv\Scripts\python.exe tools\download_weights.py
.\.venv\Scripts\python.exe tools\export_waft.py
.\.venv\Scripts\python.exe tools\export_waft_fp16.py
.\.venv\Scripts\python.exe tools\package_waft.py

$nukeExe = 'C:\Program Files\Nuke16.0v4\Nuke16.0.exe'
$env:WAFT_BUILD_VARIANT = 'FP32'
& $nukeExe --nukex -i --safe -t tools\create_cat.py
$env:WAFT_BUILD_VARIANT = 'FP16'
& $nukeExe --nukex -i --safe -t tools\create_cat.py
Remove-Item Env:\WAFT_BUILD_VARIANT
```

Replace `$nukeExe` with your Nuke 16 executable. The checkpoint is downloaded
using the recorded Google Drive ID and verified against its SHA-256. A mismatched
download does not replace the checkpoint. The source license is `WAFT-source/LICENSE`.
Intermediate `.pt` exports appear in `build/`; finished `.cat` files appear in
`WAFT/Cattery/WAFT/`.

## 4. Verify

If you skipped rebuilding, set `$nukeExe` as above first.

```powershell
.\.venv\Scripts\python.exe tools\make_test_images.py
& $nukeExe --nukex -i -t tools\verify_nuke.py
& $nukeExe --nukex -i -t tools\verify_resolution_v2.py
& $nukeExe --nukex -i -t tools\verify_devices.py
```

The first test renders Motion Vectors and Smart Vectors for known motion and
checks channels, dimensions and numerical accuracy. The second checks vector
scaling. The device check uses the installed `plugins/Cattery/WAFT/` package and
renders GPU FP16, GPU FP32, CPU FP32 and the return to GPU in one process, clearing
the RAM cache between cases. It checks graph connections and vector values after
each switch. Fresh reports are written to `build/`. After a complete export, additional
checks are available; they require CUDA and the checkpoint:

```powershell
.\.venv\Scripts\python.exe tools\verify_upstream.py
.\.venv\Scripts\python.exe tools\benchmark_precision.py FP32
.\.venv\Scripts\python.exe tools\benchmark_precision.py FP16
```

User-video scripts reference an external clip in their source. That clip is not
included; point those scripts at your own footage before rerunning them.

## 5. Package and deploy

```powershell
.\.venv\Scripts\python.exe tools\bundle_waft.py
```

This creates `WAFT-Cattery-Nuke16.zip` with the models. The packager uses fresh
reports from `build/` and falls back to saved reports in `reports/`; each report's
origin is recorded in `validation.json`. Saved reports describe the previous build.
It also regenerates `model-manifest.json` with both model sizes and SHA-256 hashes;
deploy that manifest together with the updated models.

After verification, copy the complete `WAFT/Cattery/WAFT/` package into
`../../plugins/Cattery/WAFT/` and restart Nuke. For a standalone install, copy the
same complete folder into `.nuke/Cattery/`.

## Ignored artifacts

`.gitignore` excludes environments, checkpoints, build output, backups, ZIPs,
Python caches, `.pt`, `.pth` and development `.cat` copies. Sources, licenses,
instructions and baseline reports are retained in Git. The two deployed models
under `plugins/Cattery/WAFT/` are an explicit Git LFS exception and are available
after an LFS-enabled clone. Release archives must include the actual model
binaries. The `development/` directory is unnecessary for plugin use.

On the original workstation, `D:/waft` remains intact. Heavy development copies
were moved outside Kyven to `D:/waft/kyven-cleanup-backup-20261115/`.
