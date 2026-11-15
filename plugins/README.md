# Standalone Nuke plugins

`Cattery/WAFT/` is a complete, portable Cattery plugin, including its FP16 and
FP32 `.cat` models, gizmo, callbacks, icons, license and validation reports.
Kyven registers it through `hosts/nuke/init.py` and provides
**Nodes > Kyven > WAFT SmartVector**. It runs through Nuke 16's native Inference
nodes and does not use Kyven Server or Model Manager.

To use it independently, copy the entire `WAFT` folder into
`%USERPROFILE%/.nuke/Cattery/`. Preserve its contents and layout.

The local installation includes the models. Model binaries are excluded from Git;
distribute this complete directory in a release archive, including both `.cat` files.
A source-only Git checkout does not contain the embedded models.

The lightweight development sources and [recovery instructions](../development/waft/README_RU.md)
are preserved in `../development/waft/`. Generated environments, exports and model
copies are excluded from Git and are unnecessary for deployment.
