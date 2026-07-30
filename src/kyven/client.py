"""Dependency-light client for the authenticated local Kyven server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class KyvenClientError(RuntimeError):
    """A server or transport failure safe to display in host adapters."""


@dataclass(frozen=True, slots=True)
class KyvenClient:
    """Small stdlib-only HTTP client suitable for embedded host Python."""

    base_url: str
    token: str
    timeout_seconds: float = 10.0

    @classmethod
    def from_token_file(
        cls,
        token_file: Path,
        *,
        port: int = 8765,
        timeout_seconds: float = 10.0,
    ) -> KyvenClient:
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise KyvenClientError(f"Kyven token file is empty: {token_file}")
        return cls(f"http://127.0.0.1:{port}", token, timeout_seconds)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise KyvenClientError(f"Kyven server returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise KyvenClientError(f"Could not communicate with Kyven server: {exc}") from exc

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/v1/health")

    def models(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/v1/models")
        return list(response["models"])

    def submit_segment(self, payload: dict[str, Any]) -> str:
        response = self._request("POST", "/v1/jobs/segment", payload)
        return str(response["job_id"])

    def job(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/jobs/{job_id}")

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/v1/jobs/{job_id}/cancel", {})

    def unload_all(self) -> dict[str, Any]:
        return self._request("POST", "/v1/providers/unload-all", {})

    def wait(
        self,
        job_id: str,
        *,
        poll_seconds: float = 0.2,
        timeout_seconds: float = 600.0,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            job = self.job(job_id)
            if job["status"] in {"succeeded", "failed", "cancelled"}:
                return job
            time.sleep(poll_seconds)
        raise KyvenClientError(f"Timed out waiting for Kyven job {job_id}.")
