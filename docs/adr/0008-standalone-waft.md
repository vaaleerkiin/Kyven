# 0008: Bundle WAFT as an independent Cattery plugin

Status: accepted by the owner on 2026-11-15.

The owner requested a standalone WAFT SmartVector plugin alongside Kyven, with
embedded models, and the ability to copy its complete folder back into Cattery.

Keep the Cattery package intact under `plugins/Cattery/WAFT/`. Register its path
at Nuke startup and expose one command in the Kyven menu. WAFT uses Nuke's native
Inference nodes, with no Kyven server, provider or model-manager dependency. This
is an explicit exception to the server-only inference architecture for this
optional external plugin; core Segment, Refine and Inpaint remain unchanged.

Both FP16 and FP32 models remain inside the local plugin folder and complete
release bundles. Large model binaries and generated development artifacts are
excluded from Git history. Keep lightweight development sources and recovery
instructions in Git. Preserve upstream licenses and validation reports.
Hardware and CPU behavior are those of the standalone WAFT package; the core
Kyven CPU/4 GB support contract is not extended to this plugin.
