"""Maintainer command: freeze selected upstream files, never model weights."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> None:
    entries = []
    for repo in ("Tongyi-MAI/Z-Image-Turbo", "black-forest-labs/FLUX.2-klein-4B",
                 "unsloth/FLUX.2-klein-4B-GGUF"):
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true",
                                    timeout=60) as response:
            data = json.load(response)
        revision = data["sha"]
        for item in data["siblings"]:
            name = item["rfilename"]
            if repo.startswith("Tongyi"):
                if not name.startswith(("text_encoder/", "tokenizer/")):
                    continue
                relative = name
            elif repo.startswith("black-forest"):
                if not (name.startswith(("scheduler/", "vae/"))
                        or name in {"model_index.json", "transformer/config.json"}):
                    continue
                relative = name
            elif name == "flux-2-klein-4b-Q4_K_M.gguf":
                relative = name
            else:
                continue
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
            checksum = item.get("lfs", {}).get("sha256")
            if not checksum:
                with urllib.request.urlopen(url, timeout=60) as response:
                    checksum = hashlib.sha256(response.read()).hexdigest()
            entries.append({"path": relative, "url": url, "sha256": checksum,
                            "size_bytes": item["size"], "repo": repo, "revision": revision})
    result = {"license": "Apache-2.0", "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
              "inference_revision": "2.4.0", "files": entries}
    path = Path(__file__).resolve().parents[1] / "src/kyven/resources/generative_fill/models.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Locked {len(entries)} files, {sum(x['size_bytes'] for x in entries):,} bytes")


if __name__ == "__main__":
    main()
