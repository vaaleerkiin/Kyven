# Development benchmarks

These measurements validate the current vertical slice; they are not release guarantees.

## SAM 2.1 Small - Windows development machine

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB VRAM
- Runtime: PyTorch 2.10.0 + CUDA 12.8
- Model: SAM 2.1 Hiera Small, BF16 autocast
- Input: official SAM 2 truck fixture, 1,800 x 1,200
- Prompt: one positive point

| Measurement | Result |
| --- | ---: |
| Cold in-process inference | 3.77 s |
| Warm in-process inference | 0.10 s |
| Warm server job including polling | 0.44 s |
| Peak CUDA memory allocated | 598 MiB |
| Peak CUDA memory reserved | 654 MiB |
| Selected-mask score | 0.883 |

The cold CLI process additionally pays Python, CUDA, and process startup costs. A persistent Kyven
Server keeps the active model resident, which is why host-driven repeat jobs are faster.

## Processing ROI validation

The current API 4 ROI path was verified on the Nuke development fixture:

| Item | Result |
| --- | ---: |
| Source dimensions | 1,334 x 720 |
| ROI sent to SAM 2.1 Small | 739 x 359 |
| Restored image matte | 1,334 x 720 |
| Restored three-frame tracking mattes | 3 x 1,334 x 720 |

SAM 2 resizes source data to a fixed internal encoder resolution. These checks validate cropping,
coordinate translation, and full-frame reconstruction; they do not claim proportional GPU-memory
or runtime savings from the ROI area.
