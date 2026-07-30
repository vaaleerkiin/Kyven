# Kyven Server

Kyven Server isolates PyTorch, CUDA, model loading, and inference from Nuke, Fusion, and Resolve.
It binds only to `127.0.0.1` and requires a random bearer token on every endpoint.

```text
kyven serve \
  --models-dir models \
  --device auto \
  --port 8765 \
  --token-file .runtime/server.token
```

The token is generated atomically when missing. Host adapters read it from the configured file;
it must not be committed or logged.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/v1/health` | Authenticated readiness check |
| `GET` | `/v1/models` | Model choices, installation state, and hardware guidance |
| `POST` | `/v1/jobs/segment` | Queue a segmentation job |
| `POST` | `/v1/jobs/segment-video` | Queue SAM 2 temporal mask propagation |
| `POST` | `/v1/jobs/refine` | Queue ViTMatte alpha refinement |
| `GET` | `/v1/jobs/{id}` | Read status or result |
| `POST` | `/v1/jobs/{id}/cancel` | Request cooperative cancellation |
| `POST` | `/v1/providers/unload-all` | Safely unload models after active work |

GPU jobs execute in a single-worker queue. HTTP requests and host UIs remain responsive. Model
unloading is queued behind active inference so it cannot race a running kernel.

### Segment payload fields

| Field | Meaning |
| --- | --- |
| `source`, `output` | Absolute local paths |
| `model_id` | Stable model catalog identifier |
| `profile` | `low_memory`, `balanced`, or `quality` |
| `points` | Top-left-origin positive and negative point prompts |
| `box` | Optional model prompt box |
| `roi` | Optional preprocessing crop, separate from the model prompt |
| `multimask_output` | Ask the provider for candidate masks and select the best score |
| `fill_holes` | Enable enclosed-hole post-processing; defaults to `true` |
| `max_hole_area` | Largest filled component in pixels; `0` means unlimited |

Video jobs use the same prompt and ROI fields plus `frames_dir`, `output_pattern`, frame range,
key frame, direction, and CPU-offload options. ROI output is always reconstructed to the original
frame dimensions before the job succeeds.

Refine jobs contain absolute `source`, `mask`, and `output` paths, optional `roi`, trimap generation
settings, and tiling controls. Outside an enabled Refine ROI, the coarse mask is preserved.

API version 5 adds refinement while retaining the optional Segment ROI and enclosed-hole cleanup. The
server crops inference inputs and restores returned masks to the original dimensions. The Nuke
adapter uses versioned port `18768` to avoid connecting to stale API processes during development.
