# Troubleshooting

## Server does not become ready

1. Close Nuke completely and start it again.
2. Check `D:/Kyven/.runtime/server.log`.
3. Confirm that `D:/Kyven/.venv/Scripts/python.exe` exists.
4. In Nuke, choose `Kyven > Start Server`.

The Nuke adapter launches `python.exe -I -m kyven.server.bootstrap`. On Windows the bootstrap clears
the DLL directory inherited from Nuke before importing PyTorch; this prevents the common
`c10.dll` / `WinError 1114` startup failure.

Kyven API 3 uses port `8767`. Older development servers may remain on 8765 or 8766, but the adapter
asks authenticated older servers to unload their models before starting API 3.

## CUDA or model loading fails

Run these commands outside Nuke:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.\.venv\Scripts\kyven.exe models list --models-dir models
```

Use SAM 2.1 Tiny or Small on an 8 GB GPU. If `torch.cuda.is_available()` is false, verify the NVIDIA
driver and reinstall the runtime from `requirements/runtime-cu128.txt`.

The optional SAM 2 compiled extension may emit a warning that mask-hole post-processing is skipped.
Core segmentation and propagation still run, but that post-processing feature is unavailable.

## Processing ROI fails

- Keep every positive point inside the enabled ROI.
- Reset the ROI if it no longer overlaps the input.
- Reprocess after changing ROI; old cached mattes are not transformed automatically.
- Remember that Nuke Viewer Y coordinates are converted to top-left image coordinates by the host
  adapter.

## Read or cache problems

The panel displays the exact cache directory for the current node. Use `Create Matte Read` only
after a frame, range, or tracking job succeeds. If files were deleted externally, process again.

`Delete Node Cache` affects one UUID folder. `Delete All Cache` affects only
`.runtime/nuke_cache`; it does not remove models, `.venv`, the authentication token, or source media.

## Useful files

| Path | Purpose |
| --- | --- |
| `.runtime/server.log` | Latest server startup and inference output |
| `.runtime/server.token` | Local bearer token; do not share or commit it |
| `.runtime/nuke_cache/` | Exported frames and generated mattes |
| `models/` | Verified model checkpoints |

When reporting a failure, include the Status text, the latest `server.log`, Nuke version, selected
model, and GPU model. Do not include private footage.
