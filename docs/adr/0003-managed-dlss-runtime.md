# ADR 0003: Kyven-managed DLSS package and automatic node operation

- Status: Accepted; optional pinned community component download
- Date: 2026-11-15
- Supersedes ADR 0002's user-supplied worker and manual setup workflow

The owner requires installation on a clean workstation through Model Manager, without another
Nuke plugin's runtime. Ship a Kyven build from `native/dlss`, with static CRT and checksummed
package resources. Remove optical-flow code from this worker. Never inspect or launch workers
from `.nuke/DLSS5Live`, game folders or legacy node paths.

Model Manager installs Kyven's worker package and pinned EXR dependency into the private environment.
Proprietary NVIDIA NR is a distinct required component. The public NVIDIA SDK release directory
inspected on this date supplies SR/RR/FG binaries but not `nvngx_dlssnr.dll`. Inspection of the linked
project's actual v1.0.0 ZIP confirms that it contains the NR DLL, despite its current README saying
third-party runtimes are excluded. Its packager copies local runtime files into the release;
its installer copies them from that ZIP into `.nuke` without downloading them individually.

Offer an explicit, source/terms-reviewed community component download through Model Manager.
Pin the archive URL, exact size and SHA-256, plus the member path, size and SHA-256. Extract only
NR; never install the foreign worker, ReShade or RenoDX. The request must explicitly acknowledge
the component source before network installation. This does not establish redistribution rights or
claim that the community package is an official NVIDIA SDK.
An explicit component import is also supported. Incomplete installations never report Ready.

The owner's subsequent instruction explicitly requests bundling NR in the preview package. Include
the pinned DLL in distribution resources, the ZIP and wheel, and verify its size/hash when packaging
and installing. Model Manager installs this bundled copy without downloading the community archive.
The DLL stays outside Git model-weight commits. Source-only distributions lacking the file retain
the explicit download/import path. This packaging instruction does not change third-party ownership.

The node uses native sections consistent with Segment/Refine/Inpaint and moves vector conversion
and expert thresholds to Advanced. Live starts enabled, Source-only processing uses an automatic
per-node cache, and Auto mask uses a connected Mask input or full-frame inference. Source alpha
remains available as an explicit inference guide. Upgrade in place preserves UUID, source inputs,
parameters and cached reads. Interface versioning avoids repeatedly overriding Live preferences.
