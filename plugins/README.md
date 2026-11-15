# Standalone Nuke plugins

`Cattery/WAFT/` is a complete, portable Cattery plugin, including its FP16 and
FP32 `.cat` models, gizmo, callbacks, icons, license and validation reports.
Kyven registers it through `hosts/nuke/init.py` and provides
**Nodes > Kyven > WAFT SmartVector**. It runs through Nuke 16's native Inference
nodes and does not use Kyven Server or Model Manager.

To use it independently, copy the entire `WAFT` folder into
`%USERPROFILE%/.nuke/Cattery/`. Preserve its contents and layout.

Both deployed models are included in the repository through Git LFS. With Git LFS
installed, a normal clone checks out both models beside the gizmo. For an existing
checkout run `git lfs install` and `git lfs pull` from the repository root.
Distribute this complete directory, including both `.cat` files, in release archives.

The lightweight development sources and [recovery instructions](../development/waft/README_RU.md)
are preserved in `../development/waft/`. Generated environments, exports and model
copies are excluded from Git and are unnecessary for deployment.
