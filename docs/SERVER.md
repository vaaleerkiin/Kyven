# Kyven Tools Server

[Documentation home](README.md) · [Project scope](../PROJECT_BIBLE.md) ·
[Models](MODELS.md) · [Troubleshooting](TROUBLESHOOTING.md)

Kyven Server is the shared execution layer for Kyven. It isolates PyTorch, CUDA, model loading, and
task-specific inference from Nuke. Segment, Refine, Inpaint, and optional DLSS use the provider/job system instead of placing inference inside a host adapter. DLSS uses a separate Windows/RTX worker. The server binds only to `127.0.0.1` and
requires a random bearer token on every endpoint.

Standalone [WAFT SmartVector](WAFT.md) is registered alongside Kyven but runs
inside Nuke's native Inference graph. It does not use these endpoints, the server
job queue, Model Manager or the server cache contract.

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
| `POST` | `/v1/jobs/generative-fill` | Queue local masked generation using Mage or FLUX/LanPaint |
| `POST` | `/v1/jobs/prompt-enhance` | Improve a prompt with the shared Qwen3-4B weights |
| `POST` | `/v1/jobs/dlss` | Queue optional local DLSS reconstruction with external motion/control guides |
| `POST` | `/v1/preview/trimap` | Build trimap on CPU without ViTMatte |
| `POST` | `/v1/preview/refine-tiles` | Draw exact Refine RUN/SKIP tile plan on CPU; no model loading |
| `POST` | `/v1/preview/mask-postprocess` | Rebuild matte from raw SAM output on CPU |
| `POST` | `/v1/preview/inpaint-mask` | Build exact LaMa model and final blend masks on CPU |
| `GET` | `/v1/jobs/{id}` | Read status or result |
| `POST` | `/v1/jobs/{id}/cancel` | Request cooperative cancellation |
| `POST` | `/v1/providers/unload-all` | Safely unload models after active work |
| `POST` | `/v1/server/shutdown` | Gracefully stop the authenticated local server |

GPU jobs execute in a single-worker queue. Identical active payloads share one job ID instead of
running twice, and completed history is bounded to prevent long Nuke sessions from growing server
memory indefinitely. HTTP requests and host UIs remain responsive. Model
unloading is queued behind active inference so it cannot race a running kernel. Model removal uses
the same inference queue before deleting the exact catalog-owned file. Downloads stream to a
temporary file and activate atomically only after byte-size and SHA-256 verification.

`GET /v1/health` also reports the loaded provider IDs and queued/running/completed job counts. During
a managed cold start, Nuke reports the current startup phase and elapsed/timeout seconds in the node
Status. On Windows the launcher scans listening ports once and contacts only detected older Kyven
servers, avoiding a long sequential timeout across every historical port.

### Generative Fill

Generative Fill and prompt-enhancement payloads are documented in
[Generative Fill](GENERATIVE_FILL.md#cli-and-api). Both use the same single-job
scheduler and cancellation endpoints as the core providers. They require the
optional private runtime; inference remains outside Nuke and does not use ComfyUI.

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
| `logits_output` | Optional internal compressed float16 SAM-logit cache path |
| `trimap_output` | Optional exact black / gray / white confidence-trimap PNG path |
| `confidence_width` | Non-negative symmetric logit threshold; defaults to `1.0` |

### Video propagation fields

| Field | Meaning |
| --- | --- |
| `frames_dir`, `output_pattern` | Absolute JPEG input directory and printf-style PNG destination |
| `raw_output_pattern` | Optional printf-style unmodified SAM matte sequence |
| `first_frame`, `last_frame`, `key_frame` | Nuke frame range and prompt frame |
| `direction` | `forward`, `backward`, or `both` |
| `polish_key_frame` | When true, seed a fresh chain from the nearest completed neighbour and replace only the key matte |
| `points` | Prompts sampled on the key frame |
| `logits_output_pattern` | Internal compressed float16 SAM-logit cache sequence |
| `trimap_output_pattern`, `confidence_width` | Confidence-trimap sequence and logit threshold |
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
| `skip_known_tiles` | Default true; run only tiles containing unknown trimap pixels |

Refine uses the same tile planner for ViTMatte and MEMatte. `tile_size=0` means one Full Crop pass;
omitting the size uses the profile default (512 Low Memory, 1024 Balanced, 0 Quality). Results report
`tiles`, `tiles_total`, `tiles_skipped`, and `unknown_pixels`. Known pixels are restored exactly.

`/v1/preview/refine-tiles` accepts absolute `mask` and `output` paths plus the same crop, profile,
trimap, tiling, and skip settings. Optional `image_width`/`image_height` validate the Source format.
It writes a transparent RGBA overlay and returns those statistics, `processing_roi`, and `boxes`:
each box has `index`, `run`, and full-canvas top-left `x0/y0/x1/y1` coordinates (exclusive right/bottom).
The endpoint does not activate a provider, enqueue an inference job, or require a checkpoint.
`preview_labels` optionally enables tile labels; false is the clear square-grid default.
`trimap_bounds` returns unpadded support bounds for the immediate native Nuke grid. Prepared trimaps
are cached by content and morphology settings with a thread-safe 64 MB limit; tile size, overlap,
crop and label edits reuse them. `trimap_cache_hit` reports that reuse for diagnostics.

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

Successful Segment, Refine, and Inpaint outputs receive an atomic manifest under
`_kyven_work/manifests/<output filename>.kyven.json`; legacy adjacent sidecars remain readable.
It contains the deterministic cache key and result metadata. A repeated request validates both the
manifest and every required image/NPZ before returning a cache hit; missing or corrupt media forces
a safe recalculation.

API version 48 is the current Nuke/server contract. `GET /v1/jobs/{id}` returns `progress`
(0.0-1.0) and `progress_message`. Segment video requests may include `rois`, with exactly one
`{frame, x0, y0, x1, y1}` entry per range frame. The server crops inference inputs and restores
returned masks to the original dimensions. Restart Nuke after updating Kyven so the adapter does not
reuse Python modules from an older API revision.

## Durable result cache

Segment, Refine, Inpaint and DLSS share cache manifests and a content-addressed artifact store.
`_kyven_work/manifests` holds current result manifests, `manifests/revisions` retains request-key
revisions, and `artifacts` stores verified result bytes. Identical bytes within a work root share
an artifact. Existing sidecars remain readable. Publication uses temporary files and replacement;
artifacts and revision manifests are written before the current manifest.

Cache hits check request identity and SHA-256 of recorded required outputs. Missing, corrupt or
replaced output files are restored individually from verified artifacts without rewriting intact
files or predicting again. Switching settings A -> B -> A restores the earlier revision. Missing
current manifests can also use the matching revision. If artifacts are unavailable, inference
rebuilds the result. Keep `_kyven_work` for recovery. Private copies and revision history require
additional disk space; automatic garbage collection is not implemented.

File hashes are memoized using path, size, modification time and change time in a bounded
4096-entry index. Files changed during hashing are rejected. Refine checks results before image
decoding and trimap preparation. Unchanged Segment confidence trimaps are not rewritten on hits.
These changes add no GPU memory requirement and preserve CPU providers.

DLSS retains results under `_kyven_work/results`, independently of disposable request exports.
Publication validates recorded pixel identity before replacing the public EXR, copies only changed
bytes, preserves private results, and remaps the public manifest. Repeating the current temporal
frame with identical inputs and an unchanged worker preserves history without another prediction.
Replaying an earlier temporal chain still runs inference to reconstruct private GPU state, but
verified cached outputs are not rewritten. SAM video similarly depends on the complete tracking
request; changing its range does not reuse a serialized tracker.

Cache hits avoid model prediction, but input validation still does I/O. Nuke graph exports are
shared Cache Frames / Refresh Frames actions. Current Frame and range processing invoke
the shared preparation function automatically for missing inputs, then consume prepared snapshots.
Inpaint currently prepares masks and crops before its inference-cache check. The artifact store
recovers completed results; it cannot reconstruct missing model state. Tests use synthetic
providers: live Nuke and RTX behavior still requires runtime validation.
