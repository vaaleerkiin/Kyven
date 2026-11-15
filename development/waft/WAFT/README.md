# Standalone WAFT Cattery package

This is the lightweight shell for building the standalone package. Both model
files are restored into `Cattery/WAFT/` by `tools/prepare_workspace.py` or rebuilt
from source. In a full exported archive, both models are included.

## Install the exported package

Extract the complete `WAFT` folder to a permanent location. Add that folder to
your existing `.nuke/init.py`:

```python
import nuke
nuke.pluginAddPath('D:/WAFT')
```

Replace the path with your installation. Restart Nuke 16 and choose
**Cattery > Optical Flow > WAFT**, or Tab > WAFT. Alternatively, copy only the
complete nested `Cattery/WAFT/` folder into `.nuke/Cattery/`. Keep the gizmo,
callback module, metadata, icons, license and both models together.

## Use

Connect an image sequence and set Input Colorspace correctly. Start with
Processing Resolution 512 and GPU FP16. Choose Motion Vectors for native motion
channels or Smart Vectors for the smartvector layer. Write EXR with channels=all.
The output follows the source format and contains only vectors; copy them into
the source branch if RGB is needed. Turning off GPU selects the FP32 model.

Grid Suppression defaults to 1 and reduces fine ripple; 0 preserves raw flow.
Edge Threshold defaults to 0.25 working pixels. Smoothing may remove small real
movements. Lower working resolution reduces memory use at the cost of detail.

The finished plugin runs through native Nuke Inference nodes and does not need
external Python, a compiler, Kyven Server or Model Manager. Restart Nuke and
create a new node after updating; existing nodes retain their internal graph.

## Development

In a Kyven checkout, the deployed plugin is `plugins/Cattery/WAFT/`, with both
models versioned through Git LFS. The complete recovery guide is
`development/waft/README.md` in the repository root. Relative to this shell,
export tools are in `../tools/`, source is in `../WAFT-source/`, fresh reports
are in `../build/`, and saved baseline reports are in `../reports/`.
The packager records which reports came from the current build.

Model: WAFT-a1, five iterations, dynamic input dimensions. Source:
[Princeton WAFT](https://github.com/princeton-vl/WAFT). Checkpoint Google Drive ID:
`1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC`; download verification uses the SHA-256 in
`provenance.json`. See the package's `LICENSE-WAFT.txt`.
