# Instructions for Codex and AI Coding Agents

## Project mission

Build Kyven as a production-oriented, local toolkit for Segment, Refine, and Inpaint in Nuke.

## Read first

Before changing architecture, read:

1. `README.md`
2. `PROJECT_BIBLE.md`
3. relevant files under `docs/`

The Project Bible is authoritative unless a newer accepted architecture decision record explicitly supersedes it.

## Non-negotiable constraints

- Baseline support target: 4 GB VRAM and 16 GB RAM.
- CPU fallback must remain functional.
- Core backend must not import Nuke APIs.
- Host adapters must not implement model inference.
- Each model family must be behind a provider interface.
- Disabled providers must not be loaded.
- Long operations must be asynchronous and cancelable.
- The host UI thread must never run inference.
- Cache keys must be deterministic.
- Model downloads require checksums and license metadata.
- No telemetry or remote upload by default.

## Architecture preference

Use a layered design:

```text
Host UI → Client API → Engine → Scheduler / Resource Manager → Provider
```

Do not create shortcuts that couple a host node directly to PyTorch.

## Hardware policy

Any new inference feature must answer:

1. Can it run on CPU?
2. Can it run within the 4 GB VRAM profile?
3. Does it support tiling, offload or reduced working resolution?
4. What happens after out-of-memory?
5. Can its model be unloaded?
6. How is memory use tested?

A feature that only works on large GPUs must be optional and cannot replace the baseline provider for a core function.

## Implementation order

Prefer thin vertical slices:

1. provider interface
2. CLI invocation
3. tests
4. resource limits
5. cache
6. Nuke adapter
7. UI polish

Do not begin with a large custom Nuke interface before the CLI engine works.

## Error handling

Return structured errors with:

- error code
- user-readable message
- technical detail
- recoverability
- suggested action

Never expose a raw CUDA out-of-memory traceback as the only feedback.

## Testing expectations

Every provider requires:

- import test
- capability metadata test
- CPU smoke test when supported
- low-memory configuration test
- deterministic cache-key test
- cancellation test where applicable
- license metadata test

Use synthetic or redistributable fixtures only.

## Dependency rules

- Minimize heavy dependencies in the host process.
- Prefer running inference in a separate process.
- Pin critical versions.
- Avoid undocumented binary assumptions.
- Keep Windows and Linux in mind from the first implementation.
- Do not commit model weights.

## Code quality

- Python typing is required for public interfaces.
- Prefer small modules and explicit data models.
- Avoid global mutable state except through a controlled engine singleton/service.
- Document thread and process boundaries.
- Add an architecture decision record for major irreversible choices.

## Scope discipline

Do not add product areas beyond Segment, Refine, and Inpaint without explicit project-owner approval.

## Definition of done for a task

A task is complete only when:

- implementation works
- tests exist
- errors are handled
- documentation is updated
- low-memory impact is considered
- host independence is preserved
