# Kyven Tools for Nuke

[Documentation home](README.md) · [Install](INSTALLATION.md) · [Segment](SEGMENT.md) ·
[Refine](REFINE.md) · [Inpaint](INPAINT.md) · [Generative Fill](GENERATIVE_FILL.md) ·
[Troubleshooting](TROUBLESHOOTING.md)

This guide covers the five server-backed Kyven Nuke nodes. Each has a dedicated
guide with its inputs, controls, processing, outputs and cache behavior. The
[visual guide](NODES.md) explains all six nodes, including standalone WAFT;
the [control reference](NODE_CONTROLS.md) lists captured native controls.

**WAFT SmartVector** is an additional standalone Cattery plugin. Its native Nuke
Inference graph, embedded models and output controls are covered in [WAFT](WAFT.md).
The server, Process/Cache buttons and shared cache contract below apply to the
five server-backed nodes; render WAFT with ordinary Nuke Write nodes.

The owner-requested [optional DLSS node](DLSS.md) adds local RTX reconstruction with external
motion and alpha guides. Its live/temporal and EXR-cache behavior is documented separately.

## Choose a node

| Goal | Node | Detailed guide |
| --- | --- | --- |
| Roto one frame or track through a shot | **Kyven Segment** | [Segment](SEGMENT.md) |
| Turn a coarse mask/trimap into soft alpha | **Kyven Refine** | [Refine](REFINE.md) |
| Remove an object and rebuild background | **Kyven Inpaint** | [Inpaint](INPAINT.md) |
| Edit masked content using a text instruction | **Kyven Generative Fill** | [Generative Fill](GENERATIVE_FILL.md) |
| Apply optional RTX neural reconstruction | **Kyven DLSS** | [DLSS](DLSS.md) |
| Generate motion or smart vectors from a sequence | **WAFT SmartVector** | [WAFT](WAFT.md) |

Typical roto graph:

```text
Source -------------------------> Refine Source (input 1)
Source -> Segment ---------------> Refine Mask (input 0) -> composite

Segment Output = Source + Alpha: Refine Generate Trimap from Mask ON
Segment Output = Edge Guide:     Refine Generate Trimap from Mask OFF
```

Typical paint-out graph:

```text
Source ---------------------> Inpaint Source (left input)
Paint / Segment removal mask -> Inpaint Mask (right input)
```

## Connect Nuke

Run `install.cmd`, then add the repository host folder to the user's existing `.nuke/init.py`:

```python
import nuke

nuke.pluginAddPath("D:/Kyven/hosts/nuke")
```

Replace the path with the actual repository location and restart Nuke. Kyven appears under
`Nodes > Kyven`. The installer deliberately does not edit Nuke settings. See
[Installation](INSTALLATION.md) for requirements, updates, moving the repository, and unattended
setup.

## Create and update nodes

Create nodes from:

```text
Nodes > Kyven > Segment
Nodes > Kyven > Refine
Nodes > Kyven > Inpaint
Nodes > Kyven > Generative Fill
Nodes > Kyven > DLSS (Optional RTX)
Nodes > Kyven > WAFT SmartVector
```

After updating the repository, restart Nuke and upgrade old Groups when required:

- `Kyven > Upgrade Selected Segment Node`;
- `Kyven > Upgrade Selected Refine Node`;
- `Kyven > Upgrade Selected Inpaint Node`;
- `Kyven > Upgrade Selected Generative Fill Node`;
- `Kyven > Upgrade Selected DLSS Node`.

Use **Upgrade Selected Generative Fill Node** for saved Fill Groups. Restart
Nuke first to load the updated Python modules. Mage needs one generation per frame
to prepare its new live Lab/edge patch pair; old rendered caches remain readable.
WAFT updates require its complete standalone folder and a fresh node; see [WAFT](WAFT.md).

Upgrades preserve inputs, useful controls, hidden node identity, and existing readable cache. New
artist cache fields may be added without moving or deleting a legacy UUID cache.

## Shared processing behavior

| Control | Behavior |
| --- | --- |
| **Live Current Frame** | Debounced asynchronous refresh while moving or editing |
| **Process Current Frame** | Explicitly process and replace only the playhead frame |
| **Process Frame Range** | Export the selected range and process frames sequentially |
| **Generate Current Frame / Generate Frame Range** | Generative Fill's explicit processing buttons; requires a nonempty Prompt |
| **Cancel** | Cooperatively cancel export, queue wait, or inference |

Segment additionally provides SAM 2 Forward/Backward/Both tracking. Refine,
Inpaint and Generative Fill ranges are frame-independent and can show temporal
variation. Generative Fill hides Live and keeps it off. DLSS temporal processing
requires external Motion and sequential processing; see its guide.

Explicit processing shows a native Nuke progress window with stage, percentage, ETA, and Cancel.
Live work reports through Status to avoid opening a dialog after every timeline or control change.
Manual processing automatically turns Live off and leaves it off on completion.

Each node is single-flight: repeated button presses and overlapping Live callbacks cannot run two
jobs for that node. The server also merges identical active requests into one job. A failed,
cancelled, or interrupted operation releases the node's Busy state.

## Inputs and channel handling

- Segment has one Source input.
- DLSS has Source, Motion, Depth and Mask inputs; Motion is required for temporal processing. See [DLSS setup](DLSS.md#setup).
- Refine, Inpaint and Generative Fill show **Source** on the left (input 1) and
  **Mask** on the right (input 0).
- Refine, Inpaint and Generative Fill always read Mask **alpha**. Copy an RGB-only
  mask into alpha with Shuffle/Copy upstream; there is no Alpha/Red selector.
- Masks, mattes, and trimaps are written/read as raw data so OCIO/ACES does not transform numeric
  mask values.
- Inpaint and Generative Fill convert the selected **Input Colorspace** to sRGB and back afterward; cache Reads
  remain raw to avoid a second transform.

## Processing ROI

Processing ROI is an optimization crop, not a final output crop. Kyven crops the inference input,
translates prompts/masks, runs the selected provider, and restores the result to the full Source
format. Invalid or empty ROIs safely fall back to the full frame.

Segment can animate ROI during tracking. Refine, Inpaint and Generative Fill support Auto/Manual/Full crop behavior
described in their node guides. A smaller ROI can improve effective detail and reduce work, but model
internals mean performance and VRAM do not always scale directly with crop area.

## Artist cache

All five nodes use the same Folder / Name / Version layout. Segment, Refine, Inpaint and Generative Fill write PNG sequences; DLSS writes float EXR sequences:

```text
<Cache Folder>/
└── <Name>/
    └── <Version>/
        ├── <Name>_<Version>.1001.png  (DLSS: .exr)
        ├── <Name>_<Version>.1002.png  (DLSS: .exr)
        └── _kyven_work/
```

Name is labelled **Roto Name**, **Refine Name**, **Inpaint Name**, **Fill Name**, or **Name** on DLSS. The UUID is not part of the
public path. Two nodes with the same Folder, Name, and Version intentionally share and may overwrite
the same sequence.

The version root contains the public result sequence and `_kyven_work/`. Private data is organized
below that directory:

```text
_kyven_work/
├── source/       exported frames and the Segment tracking pool
├── manifests/    deterministic cache metadata
├── raw masks
├── logits and confidence trimaps
└── model-specific masks, previews, and generated patches
```

Current Frame and Frame Range write the same public sequence for each node; Segment tracking writes to its sequence too. Changing Cache
Folder, Name, or Version reconnects an existing matching sequence. This lets another application or
artist read `<Name>_<Version>.%04d.png` (or `.exr` for DLSS) without understanding Kyven internals.

Cache actions:

- **Create Matte Read** creates a normal Read for Segment or Refine.
- **Create Result Read** creates a normal Read for Inpaint, Generative Fill or DLSS.
- DLSS also provides **Load Cache** to reconnect a previously rendered sequence.
- **Delete Node Cache** removes only the selected Name/Version after confirmation.
- **Delete All Kyven Cache** appears only on legacy UUID-cache nodes and affects only
  `.runtime/nuke_cache`; it never deletes artist-selected cache roots or models.

The delete actions above apply to Segment, Refine, Inpaint and Generative Fill;
consult the [DLSS cache guide](DLSS.md#resolution-memory-and-cache) for DLSS.
Successful outputs are atomic. Manifests include the source/prompt/settings/model identity and
validate every required file. Independent ranges reuse valid frames after interruption. Incomplete
SAM 2 temporal state is restarted even if individual mask files exist.

## Outputs

Output selection uses native Nuke branches and never reruns a model.

- Segment: Source + Alpha, Alpha, Cutout, Source + Edge Guide, Edge Guide, Bypass.
- Refine: Refined Matte, Source + Refined Alpha, Refined Cutout, Trimap,
  Source + Trimap Alpha, Trimap Cutout, Source (Bypass), Tile Preview.
- Inpaint: Result, Result + Mask Alpha, Result Premult, Generated Patch, Difference, Source.
- Generative Fill: the same six output labels as Inpaint, but Result preserves
  Source alpha/AOVs; mask modes use result permission alpha, not model-only growth.
- DLSS: Result (reconstructed RGB with source alpha and auxiliary channels), Source.

See the [Segment](SEGMENT.md), [Refine](REFINE.md), [Inpaint](INPAINT.md),
[Generative Fill](GENERATIVE_FILL.md) and [DLSS](DLSS.md) guides for exact output
semantics.

## Models and server controls

**Model Manager...** installs or removes only trusted catalog models and their pinned runtimes.
Downloads activate only after exact size and SHA-256 verification. **Refresh Models** updates the
node list and hardware advice. See [Models and safety](MODELS.md).

The adapter starts a hidden external Python process on `127.0.0.1:18788` and requires API 49. A
random bearer token in `.runtime/server.token` protects every request. Nuke never imports PyTorch,
CUDA, or model implementations.

Each node exposes **Start Server** and **Stop Server** near Status. Use Stop followed by Start after
an update if a stale process remains. The newest output is written to `.runtime/server.log`.

## Current limitations

- one object per Segment node;
- partially completed SAM 2 tracking memory cannot yet resume;
- Refine, Inpaint and Generative Fill ranges are frame-independent;
- the Nuke adapter is currently developed and tested primarily on Windows;
- node layout and API may still change during pre-alpha.

Continue with the [visual guide](NODES.md) or the detailed node guides. For
failures, go directly to [Troubleshooting](TROUBLESHOOTING.md).

## Input preparation and processing

Each node keeps one working format for inputs:

- Segment: one high-quality JPEG per frame, shared by single-frame segmentation and SAM2 tracking.
- Refine: uncompressed TIFF for source and mask.
- Inpaint: uncompressed TIFF for source and masks.
- Generative Fill: the shared Inpaint TIFF preparation for Source and masks,
  with its own prompt, sampling-mask and result-permission settings.
- DLSS: one multilayer 32-bit float EXR per frame for HDR source, alpha and selected
  external guides, with lossless ZIP compression.

Segment JPEG replaces the former lossless PNG source. JPEG is lossy even at maximum quality;
this avoids duplicate exports and uses the format required by SAM2's video loader. Public matte
and inpaint results still use PNG: these are results, not duplicate input exports.

**Frame Scope** defaults to **Current Frame** for all five server-backed nodes.
Only the current-frame processing button is visible. Select **Frame Range**
explicitly to reveal the range button and bounds. Cache Frames and Refresh Frames
follow the same selection. Existing nodes receive this safe default when upgraded;
subsequent explicit selections are preserved. Cache Frames fills missing/invalid
inputs; **Refresh Frames** updates snapshots after upstream edits. **Process Current Frame**
(and Live Current Frame) fills only its missing input snapshots automatically, then submits
inference. Ready snapshots are reused. **Process Frame Range** and tracking call the same preparation
function as Cache Frames when inputs are missing or invalid, then submit inference only after
the complete selected range is ready. Existing valid frames are never exported again. Preparation
shows progress and supports cancellation; a failed or cancelled preparation does not start inference.

Input inventory uses `_kyven_work/inputs.sqlite3`, with legacy JSON receipt compatibility.
Segment uses one `source/tracking_pool` independent of the selected range. Extending a range
adds only new frames. SAM2 gets a temporary view using hard links on the cache volume, with
copy fallback on unsupported filesystems. Verified older range JPEGs can be adopted without
rendering again. New paths replace older PNG/mask exports without deleting previous files.

Inpaint model-mask setting changes invalidate prepared snapshots. Upstream pixel edits require
Refresh Frames. Preparation uses Nuke's main thread and cancellable progress; completed frames
survive cancellation. Existing nodes get controls through the normal node upgrade action.

DLSS packs Source RGB/alpha, selected motion components, depth and the active input mask
into `_kyven_work/inputs/input.<frame>.exr`. The server decodes that file once per request
and shares its channels across the guide readers. Motion/depth channel selections and
guide connections invalidate this snapshot; reconstruction settings such as Intensity reuse it.
Only needed channels are exported, without half-float conversion or lossy compression.
Existing DLSS nodes receive the new internal graph on reopen or Upgrade Selected DLSS Node.
Older separate snapshots are rebuilt in the new format on the next preparation; older files
are retained. Public result sequences and Source alpha/AOV preservation are unchanged.


### Hover help

Press **Tab** in the Node Graph and search **Start Kyven Server** or **Stop Kyven Server**
to control the shared backend without creating a node. The same actions are under Nodes > Kyven
and beside node Status. Startup reports progress on selected Kyven nodes, waits up to 120 seconds
for a cold Python/GPU initialization, and reports completion or the actual failure. Repeated starts
share one launch. Stop cancels a pending launch; failed/timed-out children are cleaned up before retry.
The log is `<Kyven>/.runtime/server.log` (or the configured runtime folder).

Source checkouts launch their current backend directly, so updating the Nuke plugin with Git/ZIP
cannot silently select an older backend wheel in the private environment. Packaged installations
without source files use the installed backend. Server API mismatches fail explicitly.

Hover over Kyven controls for plain-English explanations of their purpose, adjustment direction, and relevant dependencies. Segment, Refine, Inpaint and DLSS include help for processing, output, input caching, and model-specific settings. New nodes receive help automatically; use the corresponding Upgrade Selected Node menu action for existing nodes, or reopen the script to run node repair. Help is local and does not load inference models or use GPU memory.

Manual Process Current Frame, Process Frame Range and Segment tracking switch Live Current Frame
off before preparing inputs or submitting work. The Live checkbox is disabled while an operation
is active. Completion, cancellation and errors unlock the checkbox without enabling Live again;
artists explicitly opt in when they want live updates. A Live-triggered operation keeps its Live
setting, while the checkbox is disabled until that operation finishes.
