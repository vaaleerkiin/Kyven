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

### Video propagation fields

| Field | Meaning |
| --- | --- |
| `frames_dir`, `output_pattern` | Absolute JPEG input directory and printf-style PNG destination |
| `first_frame`, `last_frame`, `key_frame` | Nuke frame range and prompt frame |
| `direction` | `forward`, `backward`, or `both` |
| `points` | Prompts sampled on the key frame |
| `roi` | Optional static Processing ROI |
| `rois` | Optional animated ROI: exactly one `{frame, x0, y0, x1, y1}` item per frame |
| `offload_video_to_cpu`, `offload_state_to_cpu` | Reduce persistent GPU memory use |

Animated crops are normalized to the key-frame ROI size for SAM 2 and reconstructed into each
frame's original coordinates before the job succeeds.

### Refine fields

| Field | Meaning |
| --- | --- |
| `source`, `mask`, `output` | Absolute Source, mask/trimap, and refined-alpha paths |
| `trimap_output` | Optional absolute path for the exact normalized/generated trimap PNG |
| `roi` | Optional Processing ROI |
| `generate_trimap` | Generate three-state guidance from a coarse mask when true |
| `foreground_radius`, `background_radius` | Erosion and dilation radii in pixels |
| `tile_size`, `tile_overlap` | ViTMatte memory/performance controls |

Outside an enabled Refine ROI, the coarse alpha is preserved in the refined result. The persisted
trimap is black outside the ROI because those pixels were not sent to ViTMatte.

API version 7 adds optional persisted `trimap_output` for refinement. `GET /v1/jobs/{id}` returns
`progress` (0.0-1.0) and `progress_message`. A video request may include `rois`, with exactly one
`{frame, x0, y0, x1, y1}` entry per range frame. The server crops inference inputs and restores
returned masks to the original dimensions. The Nuke adapter uses versioned port `18770` to avoid
connecting to stale API processes during development.
