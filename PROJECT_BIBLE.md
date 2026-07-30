# Kyven Project Bible

## 1. Product identity

### Working name

**Kyven**

Possible ecosystem naming:

- Kyven Utils
- Kyven Segment
- Kyven Refine
- Kyven Engine
- Kyven SDK
- Kyven Manager
- Kyven for Nuke
- Kyven for Fusion

### One-sentence definition

Kyven is a local, modular and hardware-accessible AI masking framework that behaves like a native part of a professional node-based compositing pipeline.

### Product promise

An artist must be able to generate a useful mask, interrupt the AI pipeline, correct the result with ordinary compositing tools, and resume AI refinement without being forced into a proprietary object manager or a closed workflow.

### Product category

Kyven is not only an autorotoscoping plug-in. It is a reusable inference and image-processing framework for compositing hosts.

---

## 2. Non-negotiable principles

### 2.1 The graph remains in control

Kyven follows the philosophy of node-based compositing.

Every AI stage must be:

- optional
- independently executable
- externally connectable
- inspectable
- cacheable
- replaceable
- interruptible by native host nodes

A valid workflow must be:

```text
Source
  ↓
Kyven Segment
  ↓
Roto / Erode / Blur / Merge / Expression
  ↓
Kyven Refine
  ↓
Final matte
```

The user must never be forced to keep all processing inside one opaque node.

### 2.2 One node, one object

The public node workflow should use one object per node.

This avoids:

- object managers
- nested object lists
- object IDs exposed to the artist
- complicated per-object interfaces
- fragile internal state

Efficiency is recovered in the backend by grouping compatible jobs and sharing models or image embeddings where safe.

### 2.3 Low hardware is a supported target

Kyven must officially support a baseline machine with:

- 4 GB GPU VRAM
- 16 GB system RAM
- modern x86-64 CPU
- SSD strongly recommended

This does not mean identical speed or maximum resolution. It means the user can complete the core segmentation and refinement workflow without crashes or manual source-code changes.

Low-memory support is a product requirement, not a later optimization.

### 2.4 Scale without changing the project

The same project file and node setup should be able to run on:

- CPU-only fallback
- 4 GB laptop GPU
- 8–12 GB consumer GPU
- high-end workstation GPU
- optional studio inference worker

Quality should remain as consistent as practical. Performance, tiling and precision may change according to the selected execution profile.

### 2.5 Local-first and privacy-respecting

No footage, prompts, masks, telemetry or project metadata may leave the machine by default.

Any future remote or studio worker must be:

- explicitly configured
- visible in the UI
- disabled by default
- documented clearly

### 2.6 Models are replaceable adapters

The architecture must not be named or structured around a single model family.

Correct abstraction:

```text
SegmentationProvider
RefinementProvider
PropagationProvider
```

Incorrect abstraction:

```text
SAMNodeCore
ViTMatteNodeCore
```

Initial providers may use SAM-family segmentation and ViTMatte-style refinement, but public APIs must remain model-neutral.

### 2.7 No hidden resource use

If segmentation is disabled, the segmentation model must not consume VRAM.

If refinement is disabled, the refinement model must not consume VRAM.

The user must be able to:

- inspect loaded models
- unload individual models
- unload all models
- cancel queued jobs
- clear caches

---

## 3. Primary users

### Junior and independent artists

Need professional masking without a high-end workstation or subscription-only cloud service.

### Professional compositors

Need an AI tool that respects ordinary graph workflows, external roto corrections and shot-based caching.

### Small studios

Need local deployment, predictable file paths, batch processing and the ability to scale onto shared hardware later.

### Technical directors and developers

Need a documented backend API and model adapter system that can be integrated into different hosts.

---

## 4. Product scope

### Version 1 scope

- Nuke integration
- segmentation from point and box prompts
- positive and negative prompts
- single-object node workflow
- segmentation-only mode
- refinement-only mode
- combined pipeline mode
- refinement from any input mask
- automatic trimap generation
- optional external trimap input later
- still-frame and image-sequence processing
- disk cache
- asynchronous jobs
- job cancellation
- low-memory execution profiles
- model download and validation
- model unload controls
- diagnostic report

### Explicitly outside Version 1

- depth estimation
- generative fill
- relighting
- multi-user collaboration server
- cloud processing
- full spline generation
- automatic conversion of mattes into editable roto shapes
- a complex multi-object manager
- training or fine-tuning UI

### Future scope

- temporal propagation
- optical-flow-assisted cleanup
- Fusion integration
- DaVinci Resolve integration through Fusion
- remote studio workers
- command-line batch renderer
- OpenFX integration where technically and legally appropriate
- Blender compositor integration
- model marketplace or provider registry

---

## 5. User-facing node design

### Proposed universal node: `KyvenMask`

The first implementation may expose one universal node with selectable modes:

- Segment
- Refine
- Full

Long term, separate wrappers may be presented as:

- Kyven Segment
- Kyven Refine

Both approaches must call the same host-independent backend.

### Inputs

#### Input 0: Source

Required RGB or RGBA image.

#### Input 1: Mask

Optional mask used by Refine or Full mode.

Possible future input:

#### Input 2: Trimap

Optional artist-supplied trimap.

### Output

Default output is the source image with generated alpha, with an option to output matte-only.

Possible output views:

- Result
- Matte
- Base segmentation
- Trimap
- Confidence
- Difference

### Core controls

- Mode
- Provider
- Device
- Execution profile
- Precision
- Frame range
- Cache policy
- Process current frame
- Process range
- Cancel
- Unload models
- Output view

### Prompt controls

- positive points
- negative points
- bounding box
- clear prompts
- prompt frame
- propagate prompts, future

### Refinement controls

- trimap width
- foreground erosion
- background dilation
- edge softness
- preserve interior
- temporal consistency, future

---

## 6. Host-independent architecture

```text
Host Adapter
    │
    ├── Nuke Adapter
    ├── Fusion Adapter
    └── CLI Adapter
          │
          ▼
Kyven Client API
          │
          ▼
Kyven Engine
    ├── Job Scheduler
    ├── Resource Manager
    ├── Provider Registry
    ├── Cache Manager
    ├── Image I/O Layer
    ├── Diagnostics
    └── Model Providers
          ├── Segmentation
          ├── Refinement
          └── Propagation
```

### Hard architectural boundary

The backend must not import Nuke, Fusion or Resolve modules.

The host adapter must not contain PyTorch model logic.

This boundary is required for future host support.

---

## 7. Host integration strategy

### 7.1 Nuke first

Nuke is the reference host for Version 1.

Initial integration should use:

- Python for menus, panels, node creation and job control
- a Group/Gizmo-style node wrapper for graph integration where practical
- temporary or cached image/mask exchange with the backend
- optional native or Blink/C++ image operations only after the Python prototype is stable

The host UI should remain responsive while jobs run.

### 7.2 Fusion and DaVinci Resolve

Fusion support is a planned second host, not a rewrite.

The intended path is:

```text
Fusion Tool / Fuse / Script
          │
          ▼
Same Kyven Client API
          │
          ▼
Same Kyven Engine
```

Because DaVinci Resolve contains the Fusion page, the first Resolve integration should target Fusion compositions inside Resolve rather than building a separate Color-page effect immediately.

Potential host surfaces:

- Fusion script for setup and job commands
- Fuse or macro-like tool for graph representation
- external Kyven Manager panel
- frame exchange through documented Fusion mechanisms

The first backend version must therefore avoid assumptions about Nuke knob types, channel names, node callbacks or file paths.

### 7.3 CLI

A minimal CLI should exist early, even before Fusion support.

Example intent:

```text
kyven segment --input shot.exr --prompts prompts.json --output mask.exr
kyven refine --input shot.exr --mask mask.exr --output matte.exr
kyven doctor
```

The CLI proves that the engine is genuinely host-independent and simplifies testing.

---

## 8. Hardware accessibility specification

### 8.1 Supported execution tiers

#### Tier 0: CPU fallback

Purpose:

- compatibility
- testing
- emergency use

Expectations:

- slow
- no VRAM requirement
- must not crash due to missing CUDA

#### Tier 1: Low-memory

Target:

- 4 GB VRAM
- 16 GB RAM

Requirements:

- one model resident at a time by default
- automatic model unloading between stages when needed
- half precision where supported
- tiled refinement
- scaled segmentation working resolution
- CPU offload where beneficial
- sequential frame processing
- conservative prefetching
- bounded RAM cache
- disk-backed cache

This tier must be part of automated testing.

#### Tier 2: Standard

Target:

- 8–12 GB VRAM
- 32 GB RAM

Capabilities:

- larger working resolution
- persistent active model
- limited frame prefetch
- faster propagation
- larger batch sizes where safe

#### Tier 3: Workstation

Target:

- 16 GB or more VRAM
- 64 GB or more RAM

Capabilities:

- multiple models resident
- higher concurrency
- larger tiles or full-resolution passes
- parallel decoding and preprocessing

#### Tier 4: Studio worker

Future optional service.

Capabilities:

- central GPU workers
- multiple artists
- queue priorities
- shared cache
- authenticated access

### 8.2 Memory governor

Kyven must include a Resource Manager that estimates memory before inference.

Responsibilities:

- detect available VRAM and RAM
- choose safe precision
- choose working resolution
- choose tile size and overlap
- choose batch size
- unload inactive models
- reject unsafe jobs with a useful explanation instead of crashing
- recover from out-of-memory errors by retrying with a lower profile

Suggested retry ladder:

```text
Full resolution / preferred precision
        ↓ OOM
Reduced batch or sequential frames
        ↓ OOM
Tiled inference
        ↓ OOM
Reduced working resolution
        ↓ OOM
CPU offload or CPU mode
```

Retries must be limited and visible in logs.

### 8.3 Quality policy

Low-memory mode may reduce speed and intermediate working resolution, but it must not silently change the final output contract.

Any quality-affecting fallback must be shown to the user and stored in metadata.

### 8.4 Model policy for low hardware

Every model provider must declare:

- minimum estimated VRAM
- supported precisions
- CPU support
- tiling support
- offload support
- recommended working resolution
- license metadata

Providers that cannot reasonably support the baseline tier may exist, but they cannot be the only provider for a core Kyven function.

---

## 9. Inference and scheduling

### Job types

- Encode source
- Segment frame
- Generate trimap
- Refine frame
- Propagate range
- Write cache
- Validate cache

### Scheduler principles

- jobs are cancelable
- identical jobs are deduplicated
- jobs sharing a source may reuse safe intermediate data
- model loading is centralized
- nodes never instantiate private copies of models
- different shots may share model residency but not image embeddings
- job state survives UI refreshes

### Processing strategies

#### Interactive

Prioritize the current frame and fast feedback.

#### Range

Process a selected frame range sequentially or in small batches.

#### By model

Group segmentation jobs, then refinement jobs, minimizing model swaps.

#### Per shot

Complete all required work for one shot before moving to another.

---

## 10. Cache design

### Goals

- deterministic reuse
- no hidden dependency on node names
- safe invalidation
- readable metadata
- portable project configuration

### Identity

Every node receives an internal UUID independent of its display name.

### Suggested structure

```text
<cache-root>/
  <project-id>/
    <source-hash>/
      <node-uuid>/
        segmentation/
        trimap/
        refinement/
        metadata.json
```

### Cache key inputs

- source identity
- source frame
- source colorspace interpretation if relevant
- crop and format
- provider ID
- model version and checksum
- prompt data
- input mask checksum
- trimap parameters
- refinement parameters
- execution settings that affect output
- Kyven engine version

### Invalidation rules

Changing manual corrections between Segment and Refine must invalidate refinement only, not segmentation.

Changing prompts must invalidate segmentation and all dependent stages.

Changing output display options must not invalidate inference.

---

## 11. API sketch

### Client operations

```text
GetCapabilities
GetDevices
GetProviders
SubmitJob
CancelJob
GetJobStatus
GetJobLog
UnloadProvider
UnloadAll
InspectCache
ClearCache
RunDiagnostics
```

### Example job request

```json
{
  "job_type": "refine",
  "source": "path-or-buffer-reference",
  "mask": "path-or-buffer-reference",
  "frame": 1012,
  "provider": "vitmatte_adapter",
  "execution_profile": "low_memory",
  "output": "cache-reference",
  "node_uuid": "..."
}
```

Transport may initially use a local process API or localhost IPC. The interface must remain serializable so a remote worker can be added later.

---

## 12. Repository structure

```text
kyven/
├── README.md
├── LICENSE
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── pyproject.toml
├── docs/
│   ├── PROJECT_BIBLE.md
│   ├── ARCHITECTURE.md
│   ├── HARDWARE.md
│   ├── CACHE.md
│   ├── HOSTS.md
│   └── PROVIDERS.md
├── src/kyven/
│   ├── api/
│   ├── engine/
│   ├── scheduler/
│   ├── resources/
│   ├── cache/
│   ├── providers/
│   │   ├── segmentation/
│   │   ├── refinement/
│   │   └── propagation/
│   ├── io/
│   ├── diagnostics/
│   └── cli/
├── hosts/
│   ├── nuke/
│   └── fusion/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── hosts/
│   └── hardware_profiles/
├── examples/
└── scripts/
```

---

## 13. Coding rules

1. Core modules must use type hints.
2. Public APIs require docstrings and tests.
3. Backend code must not import host APIs.
4. Host adapters must not import model implementation modules directly.
5. Model files are never committed to Git.
6. Model downloads require checksum validation.
7. License metadata is required for every provider.
8. No network access during inference unless explicitly configured.
9. No telemetry by default.
10. Out-of-memory errors must become structured Kyven errors.
11. Long operations must be cancelable.
12. UI callbacks must not perform inference directly.
13. Cache writes must be atomic where practical.
14. A failed frame must not corrupt completed frames.
15. Baseline low-memory tests are release blockers.

---

## 14. Licensing policy

### Project license

A permissive open-source license such as Apache 2.0 is recommended for the Kyven codebase, subject to final owner review.

### Dependency policy

Before inclusion, every dependency and model must have documented:

- license name
- source
- commercial-use status
- redistribution status
- model-weight restrictions
- attribution requirements

Avoid dependencies whose terms require the entire Kyven project to adopt an incompatible license unless that decision is explicit.

No provider may be enabled in the default distribution when its licensing status is unclear.

---

## 15. Security and privacy

- bind local services to localhost by default
- do not expose unauthenticated network ports
- validate all file paths and job payloads
- avoid arbitrary code execution through provider configuration
- store no footage in logs
- redact user paths in optional bug reports
- verify model checksums
- document where caches are stored

---

## 16. Diagnostics

`kyven doctor` should report:

- operating system
- Python version
- detected devices
- VRAM and RAM
- available providers
- model status
- cache permissions
- host adapter versions
- dependency conflicts
- recommended execution profile

The report must avoid including personal file paths unless the user explicitly requests a full report.

---

## 17. UX language

Prefer artist-facing language.

Good:

- Low Memory
- Balanced
- Maximum Quality
- Unload Models
- Process Range
- Use Input Mask

Avoid exposing unnecessary implementation terms such as tensor, CUDA graph, encoder block or attention backend in the main UI.

Advanced information may be available in diagnostics.

---

## 18. Roadmap

### Phase 0: Foundation

- repository
- license decision
- architecture boundaries
- CLI skeleton
- diagnostics
- provider interfaces
- test infrastructure

### Phase 1: Single-frame proof

- one segmentation provider
- prompt JSON format
- CLI segmentation
- output matte
- CPU and GPU execution
- low-memory profile

### Phase 2: Nuke prototype

- menu installation
- node creation
- current-frame processing
- prompt controls
- cache output
- job panel

### Phase 3: Refinement

- mask input
- trimap generation
- refinement provider
- tiled low-memory processing
- separation of segmentation and refinement caches

### Phase 4: Sequences

- frame ranges
- cancellation
- resumable cache
- progress reporting
- failed-frame recovery

### Phase 5: Production hardening

- installers
- model manager
- diagnostics export
- version migration
- Windows and Linux testing
- documentation and examples

### Phase 6: Fusion and Resolve

- Fusion host adapter prototype
- graph tool representation
- Resolve Fusion-page validation
- shared engine installation
- host-specific documentation

### Phase 7: Temporal tools

- propagation provider
- confidence and correction workflow
- temporal cleanup

---

## 19. Version 1 acceptance criteria

Kyven Version 1 is not complete until all conditions below are met:

- a user can install it without editing source files
- a 4 GB VRAM / 16 GB RAM profile can process a documented test shot
- CPU fallback works
- Nuke remains responsive during inference
- segmentation and refinement can be used as separate graph stages
- any external mask can be refined
- disabled providers consume no VRAM
- models can be unloaded without restarting Nuke
- interrupted range processing can resume from cache
- cache invalidation is documented and tested
- failed jobs return understandable errors
- all shipped model licenses are documented
- no footage leaves the machine by default

---

## 20. Final product statement

Kyven should feel less like an AI application inserted into Nuke and more like a set of native compositing operations powered by interchangeable local inference engines.

The project succeeds when an artist stops thinking about “using an AI tool” and simply uses Kyven as another reliable part of the node graph.
