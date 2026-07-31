# Kyven for Nuke

The Nuke adapter is a Group node that exports frames to the local Kyven Server and reads cached
PNG mattes back into the graph. Nuke remains responsive while server inference runs.

## Portable install

Clone or extract Kyven into its final writable directory and double-click `install.cmd`. To launch
the same installer from PowerShell, run from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

All runtime files stay inside that repository. The installer does not require administrator access,
does not alter `PATH`, and does not edit Nuke settings. It prints the exact plugin path when done.
Its console menu allows one or several SAM 2 models plus ViTMatte to be selected according to
available VRAM.
Choose the final repository location before installing; after moving it, rerun `install.ps1`.

## Connect Nuke manually

Add the repository host folder to the user's existing `.nuke/init.py`:

```python
import nuke

nuke.pluginAddPath("D:/Kyven/hosts/nuke")
```

The adapter discovers the repository from its own installed plugin path, so the folder may be placed
anywhere. `KYVEN_ROOT` is only an optional override for custom deployments. Restart Nuke and choose
`Kyven > Segment` or `Kyven > Refine` from the Nodes menu.

After updating Kyven, select an existing node and use the matching command:

- `Kyven > Upgrade Selected Segment Node` preserves its UUID, cached matte, prompts, and input;
- `Kyven > Upgrade Selected Refine Node` preserves its refined matte and adds current trimap outputs.

A newly created node always receives the latest controls and output graph.

## Live mode

Both Segment and Refine have `Live Current Frame`. When enabled, moving to another timeline frame
exports and submits that frame automatically. Only one GPU job runs at a time; rapid scrubbing does
not create concurrent model copies. Editing a Segment point or Processing ROI also invalidates and
regenerates the visible frame after a short debounce. Refine ROI changes do the same when Live is
enabled. Trimap radii and Segment mask post-process settings use separate CPU-only previews and
never rerun SAM or ViTMatte. Output-only controls do not rerun inference. Disable Live before
rendering a range.

## Segment controls

### Model and performance

- `Model` selects a SAM 2.1 checkpoint.
- `Memory Profile` records Low Memory, Balanced, or Quality intent.
- `Refresh Models` updates installation and VRAM advice from the server.

### Points

Positive points identify the object to keep. Negative points remove unwanted areas. Each point type
has an independent enable toggle, so its Viewer handles and payload can be hidden without deleting
the saved coordinates. Up to 32 controls are available per type.

### Processing ROI

The ROI is an inference crop, not a SAM box prompt:

1. Nuke exports the full source frame.
2. Kyven crops it to the enabled ROI.
3. Point coordinates are translated into crop space.
4. SAM receives only the cropped pixels.
5. Kyven places the returned mask on a full-size black canvas.

The final matte always has the source dimensions. A positive point must be inside the ROI;
negative points outside it are ignored. `Reset Points + ROI to Input` restores the ROI to the input
format and returns active points near its center.

SAM 2 resizes inputs to a fixed encoder resolution. ROI improves focus and effective detail but
does not guarantee proportional GPU-time or VRAM savings.

### Mask post-process

`Fill Enclosed Holes` removes black islands fully surrounded by the foreground mask. It runs after
SAM and after ROI reconstruction, so it does not affect inference or point coordinates. The outer
silhouette is never dilated or eroded.

`Max Hole Area (px)` limits which connected holes are filled. The default is `2048`; use a smaller
value to preserve intentional openings, or `0` to fill every enclosed hole. The Status field reports
how many holes were filled. Changing these controls changes cache identity and requires reprocessing.

### Processing modes

`Process Current Frame` creates one matte for the playhead frame.

`Process Range (Independent)` applies the same points and ROI independently to every frame between
`Range First` and `Range Last`. Use this when temporal consistency is not required or tracking is
not appropriate.

`Cancel` requests cooperative cancellation of the active job or range.

### SAM 2 video tracking

1. Move to a frame where the object is easy to identify.
2. Place points and optionally limit the Processing ROI.
3. Click `Set Current as Key`.
4. Set `Range First` and `Range Last`.
5. Choose `Forward`, `Backward`, or `Both Directions`.

Nuke exports high-quality temporary JPEG frames. Kyven creates one fresh SAM 2 tracking state,
offloads frames and state to system RAM, propagates from the key frame, and writes lossless full-size
PNG mattes. Points are evaluated explicitly on the selected key frame. An animated Processing ROI is
evaluated separately on every frame without moving the timeline or changing its keyframes; Kyven
crops each frame, normalizes the crop to the key-frame ROI size for tracking, then restores the matte
to that frame's original full-size coordinates. The native Nuke progress window shows export and
model stages, percentage, an estimated remaining time, and a Cancel control. Corrections from
multiple key frames are not implemented yet.

Points are used only to initialize the key frame and are checked only against that frame's ROI.
Animated ROIs on later frames may move away from the original point; they guide only the per-frame
crop and do not require the key-frame point to remain inside them.

## Output modes

New Segment nodes default to `Source + Alpha`.

| Mode | Result |
| --- | --- |
| `Matte` | Mask in RGB and alpha |
| `Source + Alpha` | Original RGB with the mask replacing alpha |
| `Cutout` | Source premultiplied by the generated alpha |
| `Source (Bypass)` | Unchanged input |

Changing output mode uses native Nuke nodes and never reruns SAM.

Refine has its own output list:

| Mode | Result |
| --- | --- |
| `Refined Matte` | ViTMatte alpha in RGB and alpha |
| `Source + Refined Alpha` | Original RGB with refined alpha; default |
| `Refined Cutout` | Source premultiplied by refined alpha |
| `Trimap` | Exact black / gray / white ViTMatte guidance in RGB and alpha |
| `Source + Trimap Alpha` | Original RGB with the exact trimap in alpha |
| `Trimap Cutout` | Source premultiplied by the trimap |
| `Source (Bypass)` | Unchanged input |

Trimap modes become exact after a Refine frame or range succeeds. Before the first result they show
the selected input mask channel as a useful preview. Switching modes never reruns ViTMatte.

## Cache

Each Segment node owns a UUID folder under:

```text
D:/Kyven/.runtime/nuke_cache/<node-uuid>/
```

Typical files include exported source frames, displayed `matte.%04d.png`, CPU-preview source
`raw_matte.%04d.png`, video JPEGs, `tracked_matte.%04d.png`, and
`raw_tracked_matte.%04d.png`. Refine nodes add `refine_source.%04d.png`,
`refine_mask.%04d.png`, `refined_matte.%04d.png`, exact processed trimaps, and lightweight
`trimap_preview` files under their own UUID folder.

- `Create Matte Read` creates a normal Nuke Read pointing to the current cached matte or sequence.
- `Delete Node Cache` disconnects and removes only the current node's cache after confirmation.
- `Delete All Cache` removes `.runtime/nuke_cache` after confirmation. Models and server settings
  are preserved.

## Server behavior

The adapter starts an external hidden Python process on `127.0.0.1:18772` and requires API 9. A
random token is stored in `.runtime/server.token`. Before startup, authenticated older Kyven server
revisions are asked to unload their models so they do not keep unnecessary VRAM.

Server output for the latest launch is written to `.runtime/server.log`.

## Current limitations

- one object per Segment node;
- no multi-key-frame tracking corrections yet;
- range resumption is not implemented yet;
- Refine is frame-independent and has no temporal propagation yet;
- the Nuke host adapter has been developed on Windows and still needs broader production testing.

See [Installation](INSTALLATION.md), [Troubleshooting](TROUBLESHOOTING.md), and
[Refine](REFINE.md) for the complete setup and trimap workflow.
