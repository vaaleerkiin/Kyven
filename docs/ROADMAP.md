# Kyven Tools vision and roadmap

Kyven Tools is a local, modular toolkit for node-based compositing. It is not limited to masks.
Each operation is presented as an independent node, while model loading, jobs, progress, caching,
hardware profiles, and host communication are shared by the Kyven Server.

The repository and technical interfaces keep the name `kyven`. The product and documentation use
**Kyven Tools**. Existing node names such as `Kyven Segment` and `Kyven Refine` remain valid.

## Product structure

```text
Kyven Tools
|-- Segment       available in Nuke
|-- Refine        available in Nuke
|-- Depth         planned next
|-- Inpaint       LaMa ONNX Fast and Big-LaMa Native available
|-- Utils         future focused image and paint utilities
|-- Server        shared local inference and cache service
|-- SDK           future provider and host APIs
`-- Studio        future optional shared-worker tooling
```

Every major image operation remains a separate graph node. Tools may be connected through native
host nodes, cached independently, upgraded independently, or replaced by ordinary compositing
operations. A universal opaque AI node is not planned.

## Current foundation

`Kyven Segment` and `Kyven Refine` establish the common workflow:

- local inference with no footage upload;
- selectable, license-audited models;
- Live current-frame processing and explicit range processing;
- asynchronous jobs, cancellation, progress, and readable errors;
- Processing ROI where it is technically appropriate;
- per-node disk cache, native Read creation, and cache cleanup;
- Source + Alpha as the compositing-friendly default output;
- one heavy provider resident at a time for low-memory systems;
- a 4 GB VRAM baseline target and an 8 GB recommended development target.

Future nodes should reuse these conventions rather than inventing a separate application workflow.

## Planned: Kyven Depth

Depth is intended for compositing video, not only for isolated still images. The first design uses a
hybrid workflow: a fast image model for interactive work and a temporal model for final ranges.

### Processing modes

| Mode | Intended provider | Purpose |
| --- | --- | --- |
| `Live Frame` | Depth Anything V2 Small | Fast feedback on the current frame |
| `Independent Frames` | Depth Anything V2 Small | Resumable fallback and still sequences |
| `Video Consistent` | Video Depth Anything Small | Temporally stable final range |
| `CPU Fallback` | MiDaS Tiny/OpenVINO candidate | Compatibility when CUDA is unavailable |

Model choices are candidates until they pass integration, output-quality, memory, dependency, and
license tests. Depth Anything V2 Small has 24.8 M parameters and Apache-2.0 weights. Video Depth
Anything Small has 28.4 M parameters and Apache-2.0 weights. Their Base and Large checkpoints are
CC-BY-NC-4.0 and must not be offered as commercially safe Kyven defaults. MiDaS is a CPU fallback
candidate, but its exact code and checkpoint combination must receive a separate license audit.

References:

- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- [MiDaS](https://github.com/isl-org/MiDaS)

### Proposed node controls

- `Mode`: Live Frame / Independent Frames / Video Consistent;
- `Model`: installed compatible providers only;
- `Memory Profile`: Low Memory / Balanced / Quality;
- `Processing Resolution`: Auto / 384 / 518 / 768 / 1024 / Custom;
- `Device`: Auto / CUDA / CPU;
- `Depth Type`: Relative / Metric when supported by the selected provider;
- `Near` and `Far` remapping controls;
- `Invert Depth`;
- `Temporal Context`: Low / Balanced / Quality for video providers;
- `Scene Cut Detection`;
- `Temporal Stability` as an optional post-process;
- `Process Current Frame`, `Process Range`, `Cancel`, and `Create Depth Read`;
- the standard cache location and cleanup controls.

### Proposed outputs

- Depth as a float channel and grayscale preview;
- normalized depth;
- inverse depth;
- Source + Depth Alpha;
- optional false-color visualization;
- Source (Bypass).

Depth values intended for compositing should not be quantized to 8-bit PNG. The implementation must
select a float-capable cache format and define consistent channel and normalization semantics before
the public API is finalized.

### Video strategy

`Video Consistent` is the preferred final-render mode. It should use streaming temporal inference
with bounded context rather than loading an entire shot into VRAM. Range processing must support
cancel, resume, progress, and cache reuse.

Independent-frame stabilization is a fallback, not a replacement for a video model. Its planned
post-process includes:

- shot-level normalization instead of per-frame normalization;
- scene-cut detection and history reset;
- confidence-aware, edge-preserving temporal smoothing;
- motion/occlusion rejection to reduce ghosting;
- optional overlap between temporal chunks to hide chunk boundaries.

Tiling must be treated cautiously because monocular depth needs global scene context. Reducing the
working resolution of the complete frame is preferred to unrelated tiles. If tiling is introduced,
it requires explicit seam and depth-scale tests.

### Hardware policy

- use FP16 on compatible GPUs and avoid FP32 unless requested;
- unload Segment or Refine providers before loading Depth on low-memory systems;
- provide 384/518 working resolutions for the Low Memory profile;
- restore the result to the source format without implying additional model detail;
- bound temporal context and prefetch;
- fall back to independent frames or CPU instead of crashing after OOM;
- benchmark 4 GB and 8 GB systems before publishing firm VRAM claims.

The installer should eventually let the user select image-depth, video-depth, and CPU fallback
models independently. Depth is not included in the installer until the provider implementation and
checkpoint hashes are complete.

## Kyven Inpaint

Inpaint offers OpenCV's fast Apache-2.0 LaMa ONNX model and the resolution-robust Big-LaMa Native
model. The fast provider works without GPU VRAM and keeps the 4 GB baseline intact; Big-LaMa keeps
native ROI detail and is the quality option for larger repairs. PowerPaint was evaluated but is not
planned for the portable distribution because its model bundle is roughly 15 GB.

### Current node contract

- input 0: Source;
- input 1: removal/fill Mask;
- optional future reference or clean-plate input;
- current-frame preview and explicit frame-range render;
- Processing ROI with padding and full-format restoration;
- mask grow/erode, feather, context padding, and preserve-outside controls;
- deterministic seed where the provider supports it;
- standard progress, cancel, cache, Create Read, and cleanup controls;
- result, patch/mask diagnostics, and Source (Bypass) outputs.

### Future inpaint work

1. Is the goal object removal, clean-plate generation, generative fill, or separate modes?
2. Can the selected model and weights be used commercially and redistributed or downloaded safely?
3. Does it have a useful 4 GB path, CPU fallback, tiling, or offload strategy?
4. How will video avoid flicker, texture swimming, and inconsistent generated details?
5. Should temporal inpainting be a later provider rather than post-stabilized still inpainting?
6. Which cache format preserves source color, bit depth, and alpha correctly?

Both LaMa variants are selectable in the installer. Future heavy providers remain optional and
cannot silently redefine the project's 4 GB baseline.

## Future Kyven Utils

Possible utilities include paint/cleanup assistance, depth-derived masks, normal or edge helpers,
and other focused image operations. This list is exploratory. A utility should be added only when
it has a clear compositing use, a small independent node contract, deterministic caching, and a
commercially safe implementation.

## Delivery order

1. Stabilize and document Segment and Refine.
2. Introduce a generic provider/resource boundary suitable for non-mask outputs.
3. Build Depth as a CLI/server vertical slice with float-cache tests.
4. Benchmark Depth Small providers on CPU, 4 GB, and 8 GB profiles.
5. Add the `Kyven Depth` Nuke node, Live mode, video range mode, and Read creation.
6. Evaluate Inpaint workflows and licenses before choosing a provider.
7. Build the next host adapter without moving inference into the host process.

This roadmap records intent, not a release promise. Model availability, licensing, production
quality, and measured hardware behavior may change the provider choices while the public node
workflow remains stable.
