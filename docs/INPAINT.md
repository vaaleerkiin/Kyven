# Kyven Inpaint

Kyven Inpaint removes masked content from a Source image and reconstructs the background inside a
Processing ROI. It is designed for paint and clean-up work: the model sees only the useful crop,
while Kyven restores the result to the untouched full input format.

## Nuke graph

```text
Source --------------------------> Kyven Inpaint (input 0)
Roto / Paint / Segment mask -----> Kyven Inpaint (input 1)
```

Input 0 supplies RGB and the alpha preserved by the default Result output. Input 1 defines pixels
to remove. Choose Alpha or Red with **Input 1 Channel** and use **Invert Input Mask** when the mask
convention is reversed.

## Choosing a model

| Model | Best for | Resolution behavior | Hardware |
| --- | --- | --- | --- |
| **LaMa ONNX Fast** | Live preview, small objects, CPU systems | Letterboxes the ROI into 512 × 512 | CPU; no GPU required |
| **Big-LaMa Native** | Large repairs and fine background texture | Keeps native ROI detail; pads only to a multiple of 8 | CPU or 4 GB+ GPU |

Start with LaMa ONNX while positioning the mask and ROI. Switch to Big-LaMa Native when a large ROI
loses visible texture detail at the fast model's fixed input size.

## Processing ROI

| Crop Mode | Behavior |
| --- | --- |
| **Auto** | Finds mask bounds and adds Context Padding; recommended default |
| **Manual** | Uses the animatable Viewer ROI supplied by the artist |
| **Full** | Sends the complete input format to the provider |

The ROI is an optimization crop, not a visible output crop. Kyven translates the mask into crop
coordinates, processes it, and pastes the result back into the original full-size frame.

For video, animate Manual ROI only when necessary. Auto ROI follows the mask independently on every
frame, which is convenient but may vary the context seen by a frame-independent model.

## Model mask and compositing

| Control | Default | Purpose |
| --- | ---: | --- |
| **Threshold** | 0.5 | Converts soft gray input pixels into the binary mask required by LaMa |
| **Model Mask Grow** | 12 px | Removes the old antialiased object edge from model input |
| **Edge Color Match** | 1.0 | Aligns the patch RGB to clean pixels around the mask |

Threshold affects only gray pixels. A mask containing only pure black and white looks identical at
every threshold between those two values. Lower Threshold includes more gray pixels; higher
Threshold includes fewer. Model Mask Grow then expands or erodes that binary area.

**Preview Model Mask** temporarily overrides the node output with the exact binary mask sent to
LaMa. It is opt-in: when neither the checkbox nor the matching Output mode is active, Kyven performs
no preview export or server request. This keeps mask preview work out of normal Inpaint renders.

`Preprocess Input Mask` is enabled by default. In this mode Invert, Threshold, and Model Mask Grow
prepare the model input. The clean soft input mask is always used for the default Result composite.
Disable preprocessing to bypass Invert, Threshold, and Grow; only LaMa's mandatory 0.5 binary
conversion remains. For custom edge treatment, select **Generated Patch** and combine it with Source using
ordinary Nuke mask-processing nodes and Merge.

## Processing modes

- **Live Current Frame** follows timeline and relevant control changes asynchronously.
- **Process Current Frame** renders one explicit cached result.
- **Process Frame Range** exports the selected range, then queues frames sequentially.
- **Cancel** stops queued work cooperatively.

Both LaMa providers are frame-independent. A rendered sequence may show flicker or texture swimming
when the background changes significantly. Temporal inpainting is future work, not hidden post-
stabilization in the current node.

## Outputs

| Output | Result |
| --- | --- |
| **Result** | Final reconstructed RGB with opaque alpha |
| **Result + Source Alpha** | Final RGB carrying Source alpha; default |
| **Result Premult** | Result + Source Alpha after Premult |
| **Generated Patch** | Full-format uncomposited model RGB; use with an external mask and Merge |
| **Model Mask Preview** | CPU-only binary mask sent to LaMa |
| **Difference** | Absolute change between Result and Source |
| **Source** | Unchanged bypass |

## Cache

Every Inpaint node stores exported inputs, `inpaint_result.%04d.png`, the full-format
`inpaint_patch.%04d.png`, and its clean composite mask under its UUID folder.

- **Create Result Read** creates a normal Nuke Read for the cached frame or sequence.
- **Delete Node Cache** removes only this Inpaint node's generated files.
- **Delete All Kyven Cache** clears all Kyven Nuke caches but keeps models and source media.

After updating Kyven, select an existing Inpaint Group and choose
`Kyven > Upgrade Selected Inpaint Node`. The current Cache controls are added without changing the
node UUID or deleting an existing result.

## Related documentation

- [Nuke workflow and shared controls](NUKE.md)
- [Model selection and licensing](MODELS.md)
- [Portable installation](INSTALLATION.md)
- [Troubleshooting](TROUBLESHOOTING.md)
