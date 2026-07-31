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

## Removing halos and patch edges

The model mask and final blend mask are deliberately separate:

| Control | Default | Purpose |
| --- | ---: | --- |
| **Model Mask Grow** | 12 px | Removes the old antialiased object edge from model input |
| **Blend Mask Grow** | 8 px | Covers the old fringe in the final composite |
| **Blend Feather** | 4 px | Softens the generated-to-source transition |
| **Edge Color Match** | 1.0 | Aligns the patch RGB to clean pixels around the mask |

If a bright outline remains, first confirm that the incoming mask covers the complete object, then
increase Model Mask Grow and Blend Mask Grow slightly. Increase Feather only after the old fringe is
fully covered; excessive feather can mix the unwanted object edge back into the result.

Pixels outside the blend mask remain identical to Source. Enable **Preview Processed Mask** beside
the preprocessing toggle to temporarily send the live Blend Mask to the node output. It updates on
the CPU while Threshold, Grow, or Feather is adjusted and never runs LaMa. **Model Mask Preview**
shows the exact binary mask sent to the model; **Blend Mask** shows the soft area changed by the
final composite, while **Difference** displays all RGB modifications.

`Preprocess Input Mask` is enabled by default. In this mode Invert, Threshold, Model Mask Grow,
Blend Mask Grow, and Blend Feather prepare the input. Disable it to bypass all five operations:
Kyven preserves the clean soft input mask for compositing and performs only the mandatory 0.5
binary conversion required by LaMa internally. The irrelevant controls are hidden while bypassed.
Changing these controls updates both mask previews on the CPU and never runs LaMa or opens a
progress window.

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
| **Result** | Reconstructed RGB with Source alpha; default |
| **Patch** | Reconstructed RGB premultiplied by Blend Mask |
| **Model Mask Preview** | CPU-only binary mask sent to LaMa |
| **Blend Mask** | Exact soft mask used for the final composite |
| **Difference** | Absolute change between Result and Source |
| **Source** | Unchanged bypass |

## Cache

Every Inpaint node stores exported inputs, `inpaint_result.%04d.png`, and
`inpaint_processed_mask.%04d.png` under its UUID folder.

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
