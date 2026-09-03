# Kyven Tools for Nuke

[Documentation home](README.md) · [Install](INSTALLATION.md) · [Segment](SEGMENT.md) ·
[Refine](REFINE.md) · [Inpaint](INPAINT.md) · [Troubleshooting](TROUBLESHOOTING.md)

This guide covers behavior shared by all Kyven Nuke nodes. Task-specific controls live in the three
node guides, with Segment first because roto is the primary Kyven workflow.

## Choose a node

| Goal | Node | Detailed guide |
| --- | --- | --- |
| Roto one frame or track through a shot | **Kyven Segment** | [Segment](SEGMENT.md) |
| Turn a coarse mask/trimap into soft alpha | **Kyven Refine** | [Refine](REFINE.md) |
| Remove an object and rebuild background | **Kyven Inpaint** | [Inpaint](INPAINT.md) |

Typical roto graph:

```text
Source -> Segment -----------------------> composite
   |          |
   |          +-- optional Trimap --+
   |                                 v
   +------------------------------> Refine -> composite
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
```

After updating the repository, restart Nuke and upgrade old Groups when required:

- `Kyven > Upgrade Selected Segment Node`;
- `Kyven > Upgrade Selected Refine Node`;
- `Kyven > Upgrade Selected Inpaint Node`.

Upgrades preserve inputs, useful controls, hidden node identity, and existing readable cache. New
artist cache fields may be added without moving or deleting a legacy UUID cache.

## Shared processing behavior

| Control | Behavior |
| --- | --- |
| **Live Current Frame** | Debounced asynchronous refresh while moving or editing |
| **Process Current Frame** | Explicitly process and replace only the playhead frame |
| **Process Frame Range** | Export the selected range and process frames sequentially |
| **Cancel** | Cooperatively cancel export, queue wait, or inference |

Segment additionally provides SAM 2 temporal Forward/Backward/Both tracking. Refine and Inpaint are
frame-independent. Their range renders can show temporal variation because each frame is inferred
without memory.

Explicit processing shows a native Nuke progress window with stage, percentage, ETA, and Cancel.
Live work reports through Status to avoid opening a dialog after every timeline or control change.
Disable Live before starting a range or tracking pass.

Each node is single-flight: repeated button presses and overlapping Live callbacks cannot run two
jobs for that node. The server also merges identical active requests into one job. A failed,
cancelled, or interrupted operation releases the node's Busy state.

## Inputs and channel handling

- Segment has one Source input.
- Refine and Inpaint show **Source** on the left and **Mask** on the right.
- Refine/Inpaint can read a mask from Alpha or Red.
- Masks, mattes, and trimaps are written/read as raw data so OCIO/ACES does not transform numeric
  mask values.
- Inpaint converts the selected **Input Colorspace** to sRGB for LaMa and back afterward; cache Reads
  remain raw to avoid a second transform.

## Processing ROI

Processing ROI is an optimization crop, not a final output crop. Kyven crops the inference input,
translates prompts/masks, runs the selected provider, and restores the result to the full Source
format. Invalid or empty ROIs safely fall back to the full frame.

Segment can animate ROI during tracking. Refine and Inpaint support Auto/Manual/Full crop behavior
described in their node guides. A smaller ROI can improve effective detail and reduce work, but model
internals mean performance and VRAM do not always scale directly with crop area.

## Artist cache

New Segment, Refine, and Inpaint nodes share one public layout:

```text
<Cache Folder>/
└── <Name>/
    └── <Version>/
        ├── <Name>_<Version>.1001.png
        ├── <Name>_<Version>.1002.png
        └── _kyven_work/
```

Name is labelled **Roto Name**, **Refine Name**, or **Inpaint Name**. The UUID is not part of the
public path. Two nodes with the same Folder, Name, and Version intentionally share and may overwrite
the same sequence.

The version root contains only the public PNG sequence and `_kyven_work/`. Private data is organized
below that directory:

```text
_kyven_work/
├── source/       exported current/range frames and tracking range folders
├── manifests/    deterministic cache metadata
├── raw masks
├── logits and confidence trimaps
└── model-specific masks, previews, and generated patches
```

Current Frame, Frame Range, and Segment tracking write the same public sequence. Changing Cache
Folder, Name, or Version reconnects an existing matching sequence. This lets another application or
artist read `<Name>_<Version>.%04d.png` without understanding Kyven internals.

Cache actions:

- **Create Matte Read** creates a normal Read for Segment or Refine.
- **Create Result Read** creates a normal Read for Inpaint.
- **Delete Node Cache** removes only the selected Name/Version after confirmation.
- **Delete All Kyven Cache** appears only on legacy UUID-cache nodes and affects only
  `.runtime/nuke_cache`; it never deletes artist-selected cache roots or models.

Successful outputs are atomic. Manifests include the source/prompt/settings/model identity and
validate every required file. Independent ranges reuse valid frames after interruption. Incomplete
SAM 2 temporal state is restarted even if individual mask files exist.

## Outputs

Output selection uses native Nuke branches and never reruns a model.

- Segment: Source + Alpha, Alpha, Cutout, Source + Trimap, Trimap, Bypass.
- Refine: Source + Refined Alpha, Refined Matte/Cutout, trimap views, Source bypass.
- Inpaint: Result, Result + Mask Alpha, Result Premult, Generated Patch, Difference, Source.

See the [Segment](SEGMENT.md), [Refine](REFINE.md), and [Inpaint](INPAINT.md) guides for exact output
semantics.

## Models and server controls

**Model Manager...** installs or removes only trusted catalog models and their pinned runtimes.
Downloads activate only after exact size and SHA-256 verification. **Refresh Models** updates the
node list and hardware advice. See [Models and safety](MODELS.md).

The adapter starts a hidden external Python process on `127.0.0.1:18788` and requires API 32. A
random bearer token in `.runtime/server.token` protects every request. Nuke never imports PyTorch,
CUDA, or model implementations.

Each node exposes **Start Server** and **Stop Server** near Status. Use Stop followed by Start after
an update if a stale process remains. The newest output is written to `.runtime/server.log`.

## Current limitations

- one object per Segment node;
- partially completed SAM 2 tracking memory cannot yet resume;
- Refine and Inpaint ranges are frame-independent;
- the Nuke adapter is currently developed and tested primarily on Windows;
- node layout and API may still change during pre-alpha.

Continue with the [Segment](SEGMENT.md), [Refine](REFINE.md), or [Inpaint](INPAINT.md) guide. For
failures, go directly to [Troubleshooting](TROUBLESHOOTING.md).
