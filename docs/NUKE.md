# Kyven Segment for Nuke

The initial adapter is under `hosts/nuke`. It creates a `KyvenSegment` Group node with:

- Source input and matte-only output;
- selectable SAM 2.1 Tiny, Small, Base+, or Large model;
- Low Memory, Balanced, and Quality execution profiles;
- dynamically addable/removable positive and negative point controls;
- independent show/use toggles that hide disabled Viewer handles;
- an input-sized prompt box and input-centered initial points;
- source-inherited output format and canvas, including before the first matte is generated;
- Process Current Frame, Process Frame Range, and Cancel actions;
- in-process source export followed by asynchronous inference;
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

## Frame ranges

`Process Frame Range` exports the selected range and generates a lossless PNG matte sequence.
The same static points and box are applied to every frame in this initial implementation. The
resulting sequence is connected inside the Group as `matte.%04d.png`.

The current node outputs the generated matte for ordinary native graph composition. Prompt
tracking/animation and source-with-alpha output are follow-up work.
