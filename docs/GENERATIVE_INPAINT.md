# Kyven Generative Inpaint

`Kyven Generative Inpaint` is a separate prompt-guided node. It does not replace the fast LaMa-based
`Kyven Inpaint` node and does not load SDXL unless this node is rendered.

## Model and license

The provider is `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`. Kyven downloads only the FP16
Diffusers components from pinned revision `115134f363124c53c7d878647567d04daf26e41e` (about 7 GB).
The model is under CreativeML Open RAIL++-M. Commercial use is possible only when the user follows
that model license and its use restrictions. The installer and Nuke Model Manager require explicit
acceptance before download. Kyven's own source code remains Apache-2.0.

## Inputs and mask

- Connect **Source** and the removal **Mask** exactly as in classic Inpaint.
- Preprocess, Threshold, Invert, and Model Mask Grow produce the exact binary mask sent to SDXL.
- **Preview Model Mask** shows that mask immediately without running SDXL.
- Auto ROI crops to the mask plus Context Padding. Manual ROI is animatable; Full uses the frame.
- `Result + Mask Alpha` and `Result Premult` use the effective model mask, not Source alpha.

## Generation controls

- **Prompt** describes the replacement; **Negative Prompt** describes content to avoid.
- **Seed** is repeatable; **Randomize Seed** creates another variation.
- **Steps** trades time for convergence. Start around 20-30.
- **Guidance** controls prompt adherence. Start around 5-7.
- **Strength** stays below 1.0 so source context is preserved.
- **Preview** uses at most 768 px and 12 steps. **Final** uses up to 1024 px and all Steps.
- **Low Memory (8 GB)** enables CPU offload and is the recommended 8 GB default.

There is deliberately no automatic Live mode: SDXL is too heavy to run safely after every knob or
timeline change. Use **Process Current Frame** to iterate, then **Process Frame Range**. SDXL is not
temporal, so frames can flicker; keep a fixed Seed and expect downstream stabilization for video.
