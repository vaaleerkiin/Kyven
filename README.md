# Kyven

**Local, modular AI masking for professional compositing.**

Kyven is an open-source masking and matte-refinement framework designed for node-based compositing applications. Its first host integration targets Foundry Nuke, with a host-independent backend designed to support Blackmagic Fusion and DaVinci Resolve later without rewriting the inference system.

Kyven is not intended to be a monolithic “one-click roto” black box. Every stage can be used independently, inspected, cached, corrected with native compositing nodes, and continued later.

```text
Source
  ↓
Kyven Segment
  ↓
Native host corrections
  ↓
Kyven Refine
  ↓
Final matte
```

## Core principles

1. **Native compositing workflow**  
   AI stages must fit into the node graph rather than replace it.

2. **Low hardware barrier**  
   A supported baseline configuration is **4 GB VRAM and 16 GB system RAM**. Kyven may run more slowly on this tier, but core workflows must remain usable.

3. **Scales upward**  
   The same project must scale from a laptop to a high-end workstation or studio processing service.

4. **Local-first**  
   Footage and masks remain on the artist’s machine unless the user explicitly configures a remote backend.

5. **Host-independent backend**  
   Models, caching, scheduling and inference must not depend on Nuke, Fusion or Resolve.

6. **Replaceable models**  
   Segmentation and refinement models are adapters, not permanent architectural dependencies.

7. **No idle VRAM waste**  
   Disabled modules must not load their models. Models must be unloadable without restarting the host application.

## Planned modules

- `Kyven Segment` — prompt-based object segmentation
- `Kyven Refine` — refinement of any externally supplied mask
- `Kyven Propagate` — temporal mask propagation
- `Kyven Cache` — cache inspection and management
- `Kyven Manager` — batch processing, model residency and job control

## Initial target

The first production milestone is a Nuke workflow with:

- one object per node
- source image plus optional mask input
- segmentation-only mode
- refinement-only mode
- full pipeline mode
- point and box prompts
- disk cache
- low-memory inference
- cancelable background jobs
- no UI blocking during inference

## Status

Pre-alpha architecture and planning.

## Name

**Kyven** is the provisional project and ecosystem name. A formal trademark clearance is outside the scope of the repository and should be completed before commercial branding or registration.

## Development status

The first implemented vertical slice is `Kyven Segment`: a host-independent promptable
segmentation service with a lazy SAM 2.1 provider and a command-line adapter. Nuke, Fusion,
and Resolve adapters will call the same service instead of loading inference models inside
their UI processes.

See [`docs/SEGMENT.md`](docs/SEGMENT.md) for the current CLI contract and development setup.

## License

Kyven is intended to be distributed under Apache License 2.0. Model files are not committed
to this repository. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for provider and
runtime licensing metadata.
