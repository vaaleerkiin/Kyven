# Kyven Tools Server

Kyven Server is the shared execution layer for the Kyven Tools toolkit. It isolates PyTorch, CUDA,
model loading, and task-specific inference from Nuke, Fusion, and Resolve. Segment, Refine, and
Inpaint are implemented APIs; planned tools such as Depth must extend the provider/job system
instead of placing inference inside a host adapter. The server binds only to `127.0.0.1` and
requires a random bearer token on every endpoint.

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

`source`, `mask`, and `output` are absolute paths. `patch_output` optionally persists the full-format
uncomposited model RGB and `mask_output` optionally persists the clean soft input mask used by the
default Result composite. `crop_mode` is `auto`, `manual`, or `full`; manual mode uses `roi`, while
auto mode uses `context_padding`. `mask_grow`, `mask_threshold`, `invert_mask`, and `mask_channel`
control the binary model mask. `preprocess_mask=false` bypasses invert, threshold, and grow while
still performing the binary conversion required by LaMa. The CPU preview endpoint writes only the
exact model mask to `output`.
`edge_color_match` (0-1) corrects a local RGB offset measured in clean pixels around the generated
area, reducing visible patch boundaries without changing pixels outside the processed mask.
Empty masks return Source unchanged without loading a model.

### Generative Inpaint fields

`POST /v1/jobs/generative-inpaint` accepts all Inpaint fields and requires a
`generative_inpaint` catalog model. It additionally accepts `prompt`, `negative_prompt`, `seed`,
`steps` (1-100), `guidance_scale` (0-20), `strength` (0.01 to below 1.0), `low_memory`, and
`render_quality` (`preview` or `final`), `generation_mode` (`clean_plate` or `replace`), and
`seam_blend` (0-128 px). SDXL uses the same authoritative exported Model Mask and
source-safe ROI composite as classic Inpaint.

API version 22 adds Clean Plate prompting, inward-only seam blending, and boundary color matching.
API version 21 adds the separate optional SDXL Generative Inpaint job and pinned repository model
downloads. API version 20 makes the enabled Inpaint preprocessing mask authoritative for LaMa inference, final
RGB compositing, and Mask Alpha/Premult outputs. It also migrates Nuke groups to Source-left and
Mask-right connectors without swapping the connected media. API version 19 removed the server round
trip from Inpaint mask preview. Nuke evaluates Threshold,
Model Mask Grow, and inversion directly in the node graph and exports that exact mask for inference.
This keeps multiple Inpaint nodes independent and makes preview changes immediate. API version 18
removed the internal Blend Mask controls, made preview strictly opt-in, and added a true
uncomposited Generated Patch output, and serializes concurrent Nuke server startup. It retains the clean-input
mask bypass and
selectable fast LaMa ONNX and native-resolution Big-LaMa, edge color matching, the single-file
Source+Mask export, separate
binary model/effective composite masks, persisted masks, signed grow/erode,
aspect-preserving preprocessing, and diagnostic Nuke outputs. It
retains `/v1/preview/trimap` and `/v1/preview/mask-postprocess`, and adds
`/v1/preview/inpaint-mask`, so host controls update without
rerunning a model. It also retains detailed Segment and Refine progress stages and the API 7
persisted `trimap_output`. `GET /v1/jobs/{id}` returns `progress` (0.0-1.0) and
`progress_message`. A video request may include `rois`, with exactly one
`{frame, x0, y0, x1, y1}` entry per range frame. The server crops inference inputs and restores
returned masks to the original dimensions. The Nuke adapter uses versioned port `18785` to avoid
connecting to stale API processes during development.
