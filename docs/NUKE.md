# Kyven Segment for Nuke

The initial adapter is under `hosts/nuke`. It creates a `KyvenSegment` Group node with:

- Source input and matte-only output;
- selectable SAM 2.1 Tiny, Small, Base+, or Large model;
- Low Memory, Balanced, and Quality execution profiles;
- positive point, negative point, and box controls;
- Process Current Frame and Cancel actions;
- background Nuke source rendering;
- asynchronous server submission and result polling;
- model installation and VRAM warnings from the server catalog.

The adapter imports no PyTorch or SAM code. It starts Kyven Server as a hidden external process
when required.

## Development installation

Add the following to the user's existing `.nuke/init.py`, using the actual repository path:

```python
import nuke
nuke.pluginAddPath("D:/Kyven/hosts/nuke")
```

Restart Nuke and choose `Kyven > Segment` from the Nodes menu. Set `KYVEN_ROOT` if the checkout
is not at `D:/Kyven`.

## Current limitation

Nuke was not installed on the development machine during this slice, so the adapter has passed
Python compilation and payload tests but still requires an in-host validation pass. Interactive
viewer handles, range processing, and source-with-alpha output are follow-up work; the current
node outputs the generated matte for ordinary native graph composition.
