# Kyven Inpaint

Kyven Inpaint removes masked content from a Source image and reconstructs the background inside a
Processing ROI. It is designed for paint and clean-up work: the model sees only the useful crop,
while Kyven restores the result to the untouched full input format.

## Nuke graph

```text
Source --------------------------> Kyven Inpaint (input 1, left)
Paint / Segment mask -----------> Kyven Inpaint (input 0, right)
```

Input 1 supplies Source RGB. Input 0 defines pixels to remove and supplies the default output alpha.
The connector labels are the safest guide: Source is on the left and Mask is on the right. Choose
Alpha or Red with **Mask Input Channel** and use **Invert Input Mask** when the mask convention is
reversed.

## Choosing a model

| Model | Best for | Resolution behavior | Hardware |
| --- | --- | --- | --- |
| **LaMa ONNX Fast** | Live preview, small objects, CPU systems | Letterboxes the ROI into 512 × 512 | CPU; no GPU required |
| **Big-LaMa Native** | Large repairs and fine background texture | Keeps native ROI detail; pads only to a multiple of 8 | CPU or 4 GB+ GPU |

Start with LaMa ONNX while positioning the mask and ROI. Switch to Big-LaMa Native when a large ROI
loses visible texture detail at the fast model's fixed input size.

### Big-LaMa Refined

When Big-LaMa Native is selected, **Quality Mode** offers `Standard` and `Refined`. Refined is not a
third model: it reuses the installed `big-lama.pt` and optimizes the generator's internal features
from a coarse image pyramid back to native ROI resolution. It adds no model download.

| Control | Default | Effect |
| --- | ---: | --- |
| **Refinement Steps** | 15 | Optimization iterations at each higher-resolution scale |
| **Refinement Strength** | 1.0 | Learning-rate multiplier; reduce it if refinement over-corrects texture |
| **Refinement Scales** | 3 | Number of coarse-to-native pyramid levels, where the ROI is large enough |

Refined is intended for final-quality stills or selected hero frames. It is much slower than
Standard, particularly on CPU, and uses gradients during inference. On an 8 GB GPU, use Auto ROI
and keep the crop close to the repair. Kyven refuses a crop that exceeds the selected Memory
Profile's refinement budget instead of allowing a worker crash; reduce the ROI or use Standard.
Cancellation and progress reporting remain active during every refinement iteration.

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
| **Result Edge Softness** | 6 px | Feathers the final RGB inward to hide a hard LaMa seam |

Threshold affects only gray pixels. A mask containing only pure black and white looks identical at
every threshold between those two values. Lower Threshold includes more gray pixels; higher
Threshold includes fewer. Model Mask Grow then expands or erodes that binary area.

**Preview Model Mask** temporarily overrides the node output with the exact binary mask sent to
LaMa. This is a native Nuke branch, so Threshold, Invert, and Model Mask Grow update immediately
without a preview export, server request, or LaMa run. The same branch is exported for processing,
so the displayed mask and the model input cannot drift apart. The Output menu deliberately does not
repeat this temporary preview option.

`Preprocess Input Mask` is enabled by default. In this mode Invert, Threshold, and Model Mask Grow
prepare both the model input and the Result composite. Disable preprocessing to bypass Invert,
Threshold, and Grow; the original input mask is then used for compositing while LaMa still performs
its mandatory binary conversion. Result Edge Softness is bypassed in this mode so the original soft
mask remains exact. With preprocessing enabled, Result Edge Softness never expands outside the mask
and does not alter Preview Model Mask or the alpha in Result + Mask Alpha / Result Premult. For fully
custom edge treatment, set it to 0, select **Generated Patch**, and combine it with Source using
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
| **Result + Mask Alpha** | Final RGB carrying the effective Inpaint mask in alpha; default |
| **Result Premult** | Result + Mask Alpha after Premult |
| **Generated Patch** | Full-format uncomposited model RGB; use with an external mask and Merge |
| **Difference** | Absolute change between Result and Source |
| **Source** | Unchanged bypass |

## Cache

Every Inpaint node stores exported inputs, the exact `inpaint_model_mask.%04d.png` seen in preview,
`inpaint_result.%04d.png`, the full-format
`inpaint_patch.%04d.png`, and its effective Inpaint mask under its UUID folder.
When a Kyven Group is copied or pasted, the first cache operation detects the duplicated UUID,
assigns a new folder to that node, and disconnects inherited cached Reads. Two Inpaint nodes cannot
therefore overwrite or display each other's newly rendered result.

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
