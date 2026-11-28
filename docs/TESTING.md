# Testing Kyven

[Documentation home](README.md) · [Project scope](../PROJECT_BIBLE.md)

## Host-independent checks

Use a development environment with the core package, pytest, ruff and OpenEXR installed:

```powershell
python -m pip install -e ".[dev,dlss]"
python -m pytest -q -p no:cacheprovider
python -m ruff check src hosts tests
git diff --check
```

These checks use synthetic images/providers, temporary cache roots and fake Nuke APIs. They do
not download model weights or claim GPU inference quality. Distribution tests verify packaged
worker/model identities when complete binaries are present; LFS pointers cannot pass the complete
WAFT distribution check.

Regression coverage includes input snapshot reuse/refresh, settings changes, missing/corrupt media,
cancelled exports, corrupt SQLite recovery, preserved media, batched/concurrent operations, and the
distinction between corruption and lock/read-only/I/O errors. Refine checks cover Cattery-style
erosion/dilation averaging and 0.01/0.99 cutoffs, empty/full/soft masks, unequal and large erosion/dilation
radii, preview/model/saved-guidance equality, ROI clipping, preserved outside alpha and cache hits.
Output tests reject NaN/Infinity and invalid shapes before successful publication. Other suites cover
job cancellation, model lifecycle, artifact restoration, deterministic keys, old DLSS mappings,
interface callbacks, alpha-only legacy selectors, Link_Knob reordering and the Git-free installer contract.

## Real Nuke checks

Set the licensed executable explicitly. The test starts a fresh process, creates synthetic 12x10
images and saves a temporary script; it does not operate on the artist's open script or run inference.

```powershell
$env:KYVEN_NUKE_EXECUTABLE = 'C:\Program Files\Nuke16.0v4\Nuke16.0.exe'
python -m pytest tests/test_nuke_runtime.py -q -p no:cacheprovider
```

It checks native Refine morphology against binary and Cattery soft-mask references, exported TIFF
guidance against the backend, faint background rejection, manual gray guidance, manual ROI,
legacy alpha-only repair, Inpaint creation/colorspace links,
Segment label migration, DLSS channel migration, UUIDs and script reopen. Without the environment
variable it is skipped. It requires an available NukeX interactive license and is separate from CI.

The separate Refine integration check uses a synthetic cached matte and float input alpha. It
verifies opt-in defaults, smooth ROI blending, unchanged alpha outside the ROI, image-edge behavior,
narrow ROIs, all refined output modes, trimap independence, upgrade/reopen, identical inference
payloads, unchanged cache bytes, and no inference request while adjusting integration controls.

Refine tile tests compare sparse and complete inference with a context-dependent synthetic model
for both providers, exact known 0/1, gray pixels at seams and image borders, narrow crops, full-crop
passes, cancellation, memory errors, invalid output and cache identity. CPU preview tests compare
each displayed RUN box and its trimap against actual inference inputs for Full/Auto/Manual modes.
A fresh subprocess confirms preview imports no Torch/Transformers. Licensed Nuke checks cover
explicit modes, legacy custom-size migration, CPU HTTP preview, Viewer compositing, updated counts,
frame changes, temporary-file cleanup, reopen, and suppressed Live inference in Tile Preview.
Slider checks cover preserved animated Context Padding, live square outlines matching every planned
edge (including fractional ROI, top-left Y orientation and last-edge alignment), one mask export
across repeated same-frame edits, explicit refresh, and frame invalidation. Cache checks cover
upstream edits, immutable guidance, morphology changes and the memory bound.
The simplified selector is checked for Tiles/Full Crop only, stable sizes across profile changes,
legacy Auto migration and preserved size/overlap values when toggling Full Crop.
Show Tiles is checked for default-off behavior, return to all seven normal output modes,
dropdown synchronization, preserved payload/snapshots, stale background completion and script reopen.
Interactive preview regressions check native yellow guidance before any HTTP response, immediate
trimap-slider changes without export, correctly premultiplied colors, a stable three-row preview
block during loading/success/error, preserved Nuke RAM cache and coalesced slider-event bursts.

## Documentation capture

The control reference can be rebuilt without model inference:

```powershell
& 'C:\Program Files\Nuke16.0v4\Nuke16.0.exe' --nukex -i -t scripts/document_nodes.py
python -m pytest tests/test_documentation.py tests/test_nuke_tooltips.py -q -p no:cacheprovider
```

Use a separate terminal process: the capture script creates nodes in its in-memory
script and ignores saved model preferences. It does not open an artist's script,
download models or submit jobs. It writes `docs/NODE_CONTROLS.md` and a private
capture report under `.runtime/documentation-capture/`. Conditional controls can
appear after changing modes; the task guides describe those.

The README's photographic examples run actual inference. Their separate
[capture workflow](EXAMPLES.md#reproduce-the-capture) requires installed models
and the DLSS runtime; WAFT also requires a licensed Nuke 16 process.

## GPU and release qualification

Native graph checks do not establish quality or memory usage of SAM, Refine providers, LaMa,
WAFT or DLSS inference. Before release, run short real sequences: current frame/ranges, cancel/retry,
model switching/unloading, settings changes, cache reuse, script reopen, unavailable runtime and
unwritable output. Verify CPU paths and the 4 GB baseline separately from optional RTX DLSS.
Check WAFT motion direction and DLSS reprojection on a moving sequence. Existing benchmarks
describe their recorded configurations.

DLSS has opt-in real-runtime tools in `scripts/qualify_dlss_guides.py` and
`scripts/qualify_dlss_temporal.py`; WAFT checks are described in
[its recovery guide](../development/waft/README.md).
