# Installing WAFT in Kyven or Cattery

For GitHub **Code > Download ZIP**, extract the archive; both models are included
and Git is not required. For `git clone`, install Git LFS and run `git lfs install`
before cloning. Run `git lfs pull` from the root of an existing clone if needed.

This complete folder is portable. It includes `WAFT.cat` (GPU FP16) and
`WAFT_FP32.cat` (FP32, including CPU). In a Git checkout, obtain both binaries
with Git LFS; they must not be small text pointer files.

In Kyven, connect the `hosts/nuke` plugin path, restart Nuke 16 and choose
**Kyven > WAFT SmartVector**, or Tab > WAFT. Kyven Server is not required.

For standalone use, copy this entire `WAFT` folder into
`%USERPROFILE%/.nuke/Cattery/`. The resulting metadata path must be
`.nuke/Cattery/WAFT/cat.json`, without another nested Cattery directory.
Restart Nuke and choose **Cattery > Optical Flow > WAFT**.

See [README.md](README.md) for controls and output channels. In a Kyven checkout,
development sources and recovery instructions are in `development/waft/README.md`
relative to the repository root.
