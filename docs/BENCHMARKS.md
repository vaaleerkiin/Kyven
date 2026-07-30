# Development benchmarks

These measurements validate the current vertical slice; they are not release guarantees.

## SAM 2.1 Small — Windows development machine

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB VRAM
- Runtime: PyTorch 2.10.0 + CUDA 12.8
- Model: SAM 2.1 Hiera Small, BF16 autocast
- Input: official SAM 2 truck fixture, 1,800 × 1,200
- Prompt: one positive point

| Measurement | Result |
| --- | ---: |
| Cold in-process inference | 3.77 s |
| Warm in-process inference | 0.10 s |
| Warm server job including polling | 0.44 s |
| Peak CUDA memory allocated | 598 MiB |
| Peak CUDA memory reserved | 654 MiB |
| Selected-mask score | 0.883 |

The cold CLI process additionally pays Python, CUDA, and process startup costs. A persistent
Kyven Server keeps the active model resident, which is why host-driven repeat jobs are faster.
