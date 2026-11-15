# WAFT development

The lightweight, rebuildable source workspace and [recovery guide](waft/README_RU.md)
are under `waft/`. Source, local changes, export/package/test scripts, licenses,
baseline reports, dependency versions and provenance are retained in Git.
Virtual environments, checkpoints, model copies and generated renders are ignored.

The deployed plugin is `../plugins/Cattery/WAFT/`. Its embedded FP16 and FP32
models remain there for immediate use; development cleanup does not remove them.

The original `D:/waft` remains intact. Heavy files removed from this development
copy were moved outside Kyven to `D:/waft/kyven-cleanup-backup-20261115/`.
