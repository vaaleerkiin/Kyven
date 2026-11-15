# Kyven DLSS worker

Kyven-owned build target, installed with its checksummed runtime package through Model Manager.
This is a maintained derivative of the MIT project-authored worker by Kai(KJ),
revision `d61307ef4543c8991668730a4cab8019b3cb0096` of
[DLSS5-for-Nuke](https://github.com/KJzzzKJ/DLSS5-for-Nuke). See `LICENSE.upstream`.

Kyven changes:

- Builds `KyvenDLSSWorker.exe` directly from this repository; no Git clone during installation.
- Removes the entire DIS implementation and state. Only external/zero motion modes are accepted.
- Validates dimensions and rejects unsupported motion modes before allocation.
- Uses the static MSVC CRT, so end users need neither Visual Studio nor a C++ runtime installer.
- Provides `--version` and writes setup diagnostics to stderr, keeping stdout binary-only.
- Packages worker, project-authored shim, license and SHA-256 manifest in Python distribution resources.
- Worker 2.0 expands scalar artist masks into RGBA32F neural-control weights, initializes the
  preserved backbuffer from the current Source, and uses white to enable NR. R32F control textures
  are not interchangeable with this format. Masked calls require `control-mask=rgba-v1` in --version.
- Preserves the defined DXGI RGB channel order instead of guessing channel swaps from image means.

`scripts/qualify_dlss_guides.py` provides opt-in real-runtime A/B qualification of black/white,
split/soft guides, restored HDR and external-vector temporal frames. Ordinary tests use synthetic providers.

Build on a Windows development machine with `scripts/build_dlss_worker.ps1`. End-user installation
copies the prebuilt package through Model Manager; it never compiles code or searches another app.
The preview distribution bundles the pinned NVIDIA NR component at the owner's request. Model
Manager installs this local copy with size/SHA-256 verification. The native build preserves it in
the package manifest when present. Source-only distributions retain the separate download/import
path. This worker cannot perform inference without that NVIDIA component.
