# Model selection and safety

Kyven treats model implementations and model sizes as selectable providers. A node stores a
stable `model_id`, not a hard-coded Python class or checkpoint path.

## Segment catalog

| Model ID | Display name | Parameters | Download | VRAM guidance |
| --- | --- | ---: | ---: | ---: |
| `sam2.1-tiny` | SAM 2.1 Tiny | 38.9 M | 156 MB | 4 GB |
| `sam2.1-small` | SAM 2.1 Small | 46.0 M | 184 MB | 6 GB |
| `sam2.1-base-plus` | SAM 2.1 Base+ | 80.8 M | 324 MB | 8 GB |
| `sam2.1-large` | SAM 2.1 Large | 224.4 M | 898 MB | 12 GB+ |

The VRAM column is conservative Kyven guidance, not an upstream guarantee. A warning does not
block selection: the artist may try any installed model and later use the Resource Manager's
fallback behavior. On the tested RTX 4070 Laptop GPU with 8,188 MiB, Small is the default.

Only one segmentation model remains resident. Selecting another variant unloads the previous
one before loading the requested model.

## Commands

```text
kyven models list --models-dir models
kyven models download sam2.1-small --models-dir models
```

Downloads are written through a temporary file, checked against trusted byte size and SHA-256,
then moved atomically into the model directory. Every entry records its source and license.

## Refine catalog

`ViTMatte-S` and `ViTMatte-B` will use this same catalog and selection mechanism when Kyven
Refine is implemented. Segment and Refine selections remain independent so disabled stages do
not consume VRAM.
