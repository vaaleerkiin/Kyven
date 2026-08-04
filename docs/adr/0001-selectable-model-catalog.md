# ADR 0001: Separate selectable model variants from host nodes

- Status: Accepted
- Date: 2026-07-30

## Context

Artists have different hardware. Segment currently has four SAM 2.1 sizes, and Refine will have
multiple ViTMatte sizes. Hard-coding one checkpoint into a Nuke node would prevent
hardware choice, complicate upgrades, and couple host scripts to model implementations.

## Decision

Host nodes store a stable task-specific `model_id`. The server resolves that ID through a trusted
catalog containing model configuration, source, byte size, SHA-256, license, commercial-use
status, CPU support, and conservative VRAM guidance.

The UI displays all catalog choices. Hardware warnings inform but do not block the artist. Only
one provider for a task stays resident by default; selecting another variant unloads the previous
one before inference.

Segment and Refine have independent selections. Their public node and API contracts remain model
neutral.

## Consequences

- Tiny, Small, Base+, and Large can coexist in the installation without changing node code.
- Downloads can be verified before models become available.
- New providers require catalog metadata and tests but no host-node rewrite.
- VRAM guidance must be benchmarked and refined across supported hardware.
- A missing model produces a structured error instead of an implicit download during inference.
