<h1><a href="https://github.com/vaaleerkiin/Kyven"><img src="hosts/nuke/icons/kyven_logo.png" alt="Kyven Tools logo" width="48" align="middle"></a>&nbsp;Kyven Tools</h1>

**Local AI tools for Foundry Nuke: roto, alpha refinement, inpaint, and optional RTX neural reconstruction.**

Kyven's Segment, Refine, Inpaint and DLSS nodes keep model loading and long inference outside Nuke. These nodes communicate
with a local authenticated server and return ordinary image sequences, so rendered work remains
readable even when Kyven or its models are not running.

The bundled **WAFT SmartVector** plugin calculates optical flow through native
Nuke Inference nodes, with embedded models and no Kyven server. See [WAFT](docs/WAFT.md).

> **Status:** active pre-alpha, currently developed for Nuke on Windows.
>
> **Privacy and licensing:** inference stays on the workstation. Kyven code is Apache-2.0; shipped
> model entries are pinned, checksum-verified, and commercially permissive. Review
> [third-party notices](THIRD_PARTY_NOTICES.md) before production distribution.

## Start here

1. Clone with Git LFS installed, or extract a complete Kyven bundle into its final
   writable folder. For an existing checkout, run `git lfs install` and `git lfs pull`
   to obtain the embedded WAFT models.
2. Run `install.cmd` and choose the recommended models.
3. Add the printed plugin path to the user's `.nuke/init.py`:

```python
import nuke

nuke.pluginAddPath("D:/Kyven/hosts/nuke")
```

4. Restart Nuke and open `Nodes > Kyven`.
5. Create a node, choose **Cache Folder**, enter its **Name** and **Version**, then process.

The installer is portable: it does not require administrator access, modify the registry, or install
system-wide Python packages. Read the [installation guide](docs/INSTALLATION.md) for requirements,
updates, moving the repository, and unattended model selection.

## Nodes

**WAFT SmartVector** is also bundled locally as a standalone Nuke 16 Cattery
plugin, with its FP16 and FP32 models inside `plugins/Cattery/WAFT/`. Open
**Nodes > Kyven > WAFT SmartVector**; no Kyven server or model-manager download
is required. Both deployed models are versioned through Git LFS. The complete
folder can be copied into `.nuke/Cattery/`.
See [standalone plugins](plugins/README.md) for packaging and portability.

| Node | Inputs | Use it for | Recommended starting model |
| --- | --- | --- | --- |
| **Kyven Segment** | Source + Viewer points | Single-frame roto and SAM 2 video tracking | SAM 2.1 Small |
| **Kyven Refine** | Source + coarse mask/trimap | Soft alpha and difficult boundaries | ViTMatte Small |
| **Kyven Inpaint** | Source + removal mask | Paint-out and background reconstruction | LaMa ONNX Fast |
| **Kyven DLSS (optional RTX)** | Source + optional motion, depth and control mask | Neural reconstruction with temporal history | Kyven worker + separately licensed NVIDIA NR component |
| **WAFT SmartVector (standalone)** | Source image sequence | Motion Vectors or Smart Vectors | Embedded WAFT-a1 FP16 / FP32 |

External motion is required when DLSS temporal processing is enabled. The [DLSS guide](docs/DLSS.md) covers setup, external vectors, alpha as an inference guide, live
refresh and runtime limitations. Optional DLSS has no CPU inference provider.

A common roto flow is:

```text
Source -> Kyven Segment -> optional Kyven Refine -> compositing nodes
```

For temporal roto, set a key frame, add corrections where required, optionally enable
**Seamless Key Frame**, and run Forward, Backward, or Both Directions. For paint-out, connect the
original Source and a removal mask directly to Kyven Inpaint.

## Cache contract

The four server-backed nodes use the same artist-facing Folder / Name / Version layout. Segment, Refine and Inpaint write PNG results; optional DLSS writes float EXR results. Standalone WAFT uses ordinary Nuke Write rendering:

```text
<Cache Folder>/
└── <Name>/
    └── <Version>/
        ├── <Name>_<Version>.1001.png  (DLSS: .exr)
        ├── <Name>_<Version>.1002.png  (DLSS: .exr)
        └── _kyven_work/
```

The version root contains the public sequence. Sources, manifests, logits, trimaps, masks, and other
implementation files stay under `_kyven_work/`. Current Frame and Frame Range write the same named sequence; Segment tracking does too. Changing Cache Folder, Name, or Version reconnects a matching sequence.
DLSS uses the same Folder / Name / Version layout with float EXR output instead of PNG; see
the [DLSS cache guide](docs/DLSS.md#resolution-memory-and-cache).

## How it works

```text
Nuke Group node
      |
      | authenticated HTTP on 127.0.0.1
      v
Kyven Server -> single GPU job queue -> selected provider/model
      |
      v
Atomic artist cache -> ordinary Nuke Read
```

- Kyven's server adapters never import PyTorch or model packages. Standalone WAFT uses Nuke's built-in Inference runtime.
- The active provider remains resident between frames and jobs.
- Long operations are asynchronous, cancellable, and report progress.
- Cache manifests reject incomplete or mismatched results.
- Identical active requests share one server job instead of duplicating inference.

## Working with cached frames

In the Nuke nodes, choose **Input Frames** and click **Cache Frames** to prepare source inputs. Current-frame processing prepares missing inputs automatically; range processing prepares any missing frames before inference. After changing an upstream image, mask, or motion vector, use **Refresh Frames** for the affected scope and process again. Cached input snapshots and results are stored under the selected cache folder, name, and version. Hover over node controls for English explanations and adjustment guidance. See the [Nuke workflow](docs/NUKE.md) for details and existing-node upgrades.

## Documentation

Open the **[documentation home](docs/README.md)**. It provides reading paths for artists,
installation, troubleshooting, pipeline integration, and development.

Most-used guides:

- [Nuke workflow](docs/NUKE.md)
- [Segment](docs/SEGMENT.md)
- [Refine](docs/REFINE.md)
- [Inpaint](docs/INPAINT.md)
- [DLSS (optional RTX)](docs/DLSS.md)
- [WAFT SmartVector](docs/WAFT.md)
- [WAFT development recovery](development/waft/README_RU.md)
- [Models and hardware](docs/MODELS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Repository map

```text
Kyven/
├── src/kyven/       host-neutral engine, providers, server, and CLI
├── hosts/nuke/      Nuke adapter and Group-node builders
├── plugins/Cattery/WAFT/  portable WAFT plugin with local embedded models
├── development/waft/     lightweight WAFT sources and recovery instructions
├── docs/            user and technical documentation
├── tests/           engine, server, and adapter tests
├── models/          locally installed weights; not committed
└── .runtime/        local server state and logs; not committed
```

Kyven is pre-alpha. Treat cached renders as working assets, while node layouts and provider APIs may
still change between revisions.
