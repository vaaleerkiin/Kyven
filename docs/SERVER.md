# Kyven Tools Server

Kyven Server is the shared execution layer for Kyven. It isolates PyTorch, CUDA, model loading, and
task-specific inference from Nuke. Segment, Refine, and Inpaint extend the same provider/job system
instead of placing inference inside a host adapter. The server binds only to `127.0.0.1` and
requires a random bearer token on every endpoint.

```text
kyven serve \
  --models-dir models \
  --device auto \
  --port 8765 \
  --token-file .runtime/server.token
```

This example uses the CLI default port `8765`. The Nuke adapter launches its managed server on
`18788`; do not override that port in the Nuke workflow.

The token is generated atomically when missing. Host adapters read it from the configured file;
it must not be committed or logged.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/v1/health` | Authenticated readiness check |
| `GET` | `/v1/models` | Model choices, installation state, and hardware guidance |
| `POST` | `/v1/models/download` | Start a verified catalog model download |
| `POST` | `/v1/models/remove` | Safely unload and remove one catalog checkpoint |
| `GET` | `/v1/model-operations/{id}` | Read model-operation progress and result |
| `POST` | `/v1/model-operations/{id}/cancel` | Cancel an active model download |
| `POST` | `/v1/jobs/segment` | Queue a segmentation job |
| `POST` | `/v1/jobs/segment-video` | Queue SAM 2 temporal mask propagation |
| `POST` | `/v1/jobs/refine` | Queue ViTMatte alpha refinement |
| `POST` | `/v1/jobs/inpaint` | Queue LaMa Source + Mask object removal |
| `POST` | `/v1/preview/trimap` | Build trimap on CPU without ViTMatte |
| `POST` | `/v1/preview/mask-postprocess` | Rebuild matte from raw SAM output on CPU |
| `POST` | `/v1/preview/inpaint-mask` | Build exact LaMa model and final blend masks on CPU |
| `GET` | `/v1/jobs/{id}` | Read status or result |
| `POST` | `/v1/jobs/{id}/cancel` | Request cooperative cancellation |
| `POST` | `/v1/providers/unload-all` | Safely unload models after active work |
| `POST` | `/v1/server/shutdown` | Gracefully stop the authenticated local server |

GPU jobs execute in a single-worker queue. HTTP requests and host UIs remain responsive. Model
unloading is queued behind active inference so it cannot race a running kernel. Model removal uses
the same inference queue before deleting the exact catalog-owned file. Downloads stream to a
temporary file and activate atomically only after byte-size and SHA-256 verification.

### Segment payload fields

| Field | Meaning |
| --- | --- |
| `source`, `output` | Absolute local paths |
| `raw_output` | Optional unmodified SAM matte used for live post-process previews |
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
| `raw_output_pattern` | Optional printf-style unmodified SAM matte sequence |
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

### Inpaint fields

`source`, `mask`, and `output` are absolute paths. `model_mask` may point to a precomputed binary
mask that is supplied to LaMa. `patch_output` optionally persists the full-format uncomposited model
RGB, while `mask_output` persists the final blend alpha used for the RGB composite. `crop_mode` is
`auto`, `manual`, or `full`; manual mode uses `roi`, while
auto mode uses `context_padding`. `mask_grow`, `mask_threshold`, `invert_mask`, and `mask_channel`
control the binary model mask. `preprocess_mask=false` bypasses invert, threshold, and grow while
still performing the binary conversion required by LaMa. The CPU preview endpoint writes only the
exact model mask to `output`.
`edge_color_match` (0-1) corrects a local RGB offset measured in clean pixels around the generated
area, reducing visible patch boundaries without changing pixels outside the processed mask.
`edge_softness` (0-32 px) feathers the final RGB composite inward when preprocessing is enabled.
This removes hard LaMa patch seams while leaving every pixel outside the mask untouched. When
`preprocess_mask=false`, the original soft input mask is used directly as the blend alpha and edge
softness is not applied.
`quality_mode` is `standard` or `refined`. Refined is valid only for `big-lama-native` and accepts
`refinement_steps` (1-30), `refinement_strength` (0.1-2.0), and `refinement_scales` (2-4). These
fields participate in the deterministic cache key. The worker reports progress and checks
cancellation during each refinement iteration.
Empty masks return Source unchanged without loading a model.

API version 26 is the current Nuke/server contract. `GET /v1/jobs/{id}` returns `progress`
(0.0-1.0) and `progress_message`. Segment video requests may include `rois`, with exactly one
`{frame, x0, y0, x1, y1}` entry per range frame. The server crops inference inputs and restores
returned masks to the original dimensions. Restart Nuke after updating Kyven so the adapter does not
reuse Python modules from an older API revision.
