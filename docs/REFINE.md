# Kyven Refine

Refine converts a coarse mask into a soft alpha matte using ViTMatte while keeping model inference
outside the compositing host.

## Nuke graph

```text
Source --------------------> Kyven Refine (input 0)
Source -> Kyven Segment ---> Kyven Refine (input 1)
```

Input 0 is the original RGB image. Input 1 is normally any mask with useful alpha: Segment, Roto,
Keyer, Paint, or a corrected combination of those nodes.
`Input 1 Channel` defaults to Alpha; choose Red for a grayscale RGB mask or artist trimap.

## Trimap option

`Generate Trimap from Mask` is enabled by default and is part of Refine, not a separate node.
Foreground erosion produces definite white foreground, background dilation produces the gray
unknown band, and pixels outside the band remain definite black background. Disable the option only
when input 1 is already an artist-created black/gray/white trimap.

## Processing

- `Live Current Frame` follows the timeline asynchronously.
- `Process Current Frame` updates one cached matte.
- `Process Frame Range` exports and refines every selected frame sequentially.
- `Create Read from Current Matte` creates a standard Nuke Read for the cached frame or sequence.
- `Cancel` stops queued work cooperatively.

Processing ROI crops both image and mask before inference, then pastes the refined alpha into the
full-size coarse mask. Low Memory uses 512 px tiles, Balanced uses 1024 px tiles, and Quality uses a
full-frame pass unless a custom tile size is set. Overlap blending reduces tile seams.

## CLI

```text
kyven refine --input source.png --mask coarse.png --output alpha.png \
  --model vitmatte-small-composition-1k --profile low_memory
```

Add `--manual-trimap` when the mask path is already a trimap.
