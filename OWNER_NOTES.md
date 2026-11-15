# Kyven owner notes

Kyven is a local toolkit for Nuke with three core workflows:

- Segment creates a base mask using SAM 2.
- Refine turns a mask or trimap into a soft alpha channel using ViTMatte.
- Inpaint removes a marked area using LaMa.

These models run in a separate local process to keep Nuke responsive. Source
frames are not uploaded to external services. Results are written to ordinary
cache files and remain available after the model is unloaded.

Priorities are stable core workflows, clear errors, controlled memory use,
verified model downloads and safe cache management. Selectable model variants
are installed through Model Manager.

Optional DLSS uses Kyven's managed Windows/RTX worker. The worker and shim are
in the repository; the NVIDIA Neural Rendering component is separate from the
Git checkout. Model Manager prepares the DLSS runtime on first use.

WAFT SmartVector is an independent Cattery plugin. Its gizmo and two embedded
models are stored together under `plugins/Cattery/WAFT/`; the models use Git LFS.
It runs inside Nuke without Kyven Server or Model Manager. The entire folder can
be copied into Cattery unchanged.
