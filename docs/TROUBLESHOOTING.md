# Troubleshooting

## Portable installer fails

Double-click `install.cmd`, or run the installer from the repository root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

Python 3.10-3.13 must already be installed; Python 3.12 is preferred. If it is not detected, pass
its full path with `-PythonExe`. The installer does not need Git, administrator access, or an EXE
installer. Internet access is required for the first dependency and model download.

Choose the final repository location before installation. A Windows virtual environment contains
absolute paths, so after moving the Kyven folder, rerun `install.ps1` to rebuild or repair `.venv`.
It is safe to rerun after pulling an update; verified model files are reused. The installer targets
only legacy `kyven.exe` launchers from this repository and never terminates unrelated Python jobs.
Restart Nuke after installation so it starts the newly installed API revision.

## Server does not become ready

1. Close Nuke completely and start it again.
2. Check `D:/Kyven/.runtime/server.log`.
3. Confirm that `D:/Kyven/.venv/Scripts/python.exe` exists.
4. In Nuke, choose `Kyven > Start Server`.

The Nuke adapter launches `python.exe -I -m kyven.server.bootstrap`. On Windows the bootstrap clears
the DLL directory inherited from Nuke before importing PyTorch; this prevents the common
`c10.dll` / `WinError 1114` startup failure.

Kyven API 25 uses port `18787`. Older development servers may remain on 8765-8769 or 18768-18786,
but the adapter asks authenticated older servers to unload their models before starting API 25.

Each Kyven node provides **Stop Server** and **Start Server** next to Status. Use them after updating
the repository if Nuke is still connected to code loaded before the update. Stop affects only the
authenticated Kyven service on port `18787`; it does not terminate Nuke or unrelated Python tools.

## Refine fails or returns the coarse mask unchanged

- Connect the original RGB image to the left Source connector (input 1) and a mask/trimap to the
  right Mask connector (input 0).
- Keep `Generate Trimap from Mask` enabled for a normal binary Segment or painted matte.
- Disable it only for a true three-state trimap: black background, gray unknown, white foreground.
- Select Red as `Mask Input Channel` when the mask/trimap is stored in RGB instead of alpha.
- Increase erosion/dilation to give ViTMatte a wider unknown edge region.
- Use Low Memory (512 px tiles) when VRAM is limited.
- After updating from an older API, restart Nuke so it launches the server on port 18787.

## Trimap output is missing or shows only the input mask

- Restart Nuke after updating, select the Refine node, and run
  `Kyven > Upgrade Selected Refine Node`.
- Connect the Mask input. The CPU-only trimap preview should appear without processing a frame or loading
  ViTMatte.
- With Processing ROI enabled, black outside the ROI is expected: ViTMatte did not receive those
  pixels. The refined-alpha output still preserves the coarse mask outside the ROI.
- Confirm that `.runtime/nuke_cache/<node-uuid>/trimap_preview_r<revision>.<frame>.png` exists.

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
- During video propagation, points are read on `Key Frame`; make sure they are inside the ROI on that
  frame. The ROI itself may be animated and is sampled independently for every rendered frame.
- Reprocess after changing ROI; old cached mattes are not transformed automatically.
- Remember that Nuke Viewer Y coordinates are converted to top-left image coordinates by the host
  adapter.

## The mask contains small black holes

Enable `Fill Enclosed Holes` and reprocess the frame or range. Increase `Max Hole Area (px)` only
until the unwanted holes disappear; very large values may also fill intentional enclosed openings.
This cleanup does not expand the exterior edge. A SAM 2 warning about the unavailable optional `_C`
extension means upstream hole filling was skipped, but Kyven's own post-process still runs.

## Read or cache problems

The panel displays the exact cache directory for the current node. Use `Create Matte Read` for
Segment/Refine or `Create Result Read` for Inpaint only after processing succeeds. Both buttons copy
the active internal Read, including its frame range. If files were deleted externally, process again.

`Delete Node Cache` affects one UUID folder. `Delete All Kyven Cache` affects only
`.runtime/nuke_cache`; it does not remove models, `.venv`, the authentication token, or source media.

## Inpaint Result has a rectangular color shift

1. Upgrade the selected node with `Kyven > Upgrade Selected Inpaint Node`.
2. Delete that node's old cache; cached patches retain the transfer used when they were rendered.
3. Process once with **LaMa Input Color = sRGB Texture**.
4. If the ROI boundary remains under the project's OCIO configuration, choose
   **Linear / Working (Raw)** and process again. Source Write and Result/Patch Reads will then bypass
   OCIO symmetrically.

Generated Patch preserves Source RGB outside the binary model mask. If a rectangular boundary is
still visible after a fresh render, confirm that Nuke is not displaying an older cached Read.

## Useful files

| Path | Purpose |
| --- | --- |
| `.runtime/server.log` | Latest server startup and inference output |
| `.runtime/server.token` | Local bearer token; do not share or commit it |
| `.runtime/nuke_cache/` | Exported frames, generated/refined mattes, and exact trimaps |
| `models/` | Verified model checkpoints |

When reporting a failure, include the Status text, the latest `server.log`, Nuke version, selected
model, and GPU model. Do not include private footage.
