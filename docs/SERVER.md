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
| `GET` | `/v1/jobs/{id}` | Read status or result |
| `POST` | `/v1/jobs/{id}/cancel` | Request cooperative cancellation |
| `POST` | `/v1/providers/unload-all` | Safely unload models after active work |

GPU jobs execute in a single-worker queue. HTTP requests and host UIs remain responsive. Model
unloading is queued behind active inference so it cannot race a running kernel.

API version 3 accepts an optional `roi` rectangle separately from the model's `box` prompt. The
server crops inference inputs and restores returned masks to the original dimensions. The Nuke
adapter uses versioned port `8767` to avoid connecting to stale API processes during development.
