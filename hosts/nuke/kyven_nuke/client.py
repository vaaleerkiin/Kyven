"""Small Python 3.7-compatible client embedded with the Nuke adapter."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class NukeKyvenClientError(RuntimeError):
    pass


class NukeKyvenClient:
    def __init__(self, base_url, token, timeout_seconds=10.0):
        self.base_url = base_url
        self.token = token
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_token_file(cls, token_file, port=8765, timeout_seconds=10.0):
        token = token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise NukeKyvenClientError(f"Kyven token file is empty: {token_file}")
        return cls(f"http://127.0.0.1:{port}", token, timeout_seconds)

    def _request(self, method, path, payload=None, timeout_seconds=None):
        data = None
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise NukeKyvenClientError(
                f"Kyven server returned HTTP {exc.code}: {detail}"
            )
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise NukeKyvenClientError(f"Could not communicate with Kyven server: {exc}")

    def health(self):
        return self._request("GET", "/v1/health")

    def models(self):
        return list(self._request("GET", "/v1/models")["models"])

    def start_model_download(self, model_id):
        response = self._request("POST", "/v1/models/download", {"model_id": model_id})
        return str(response["operation_id"])

    def start_model_remove(self, model_id):
        response = self._request("POST", "/v1/models/remove", {"model_id": model_id})
        return str(response["operation_id"])

    def model_operation(self, operation_id):
        return self._request("GET", f"/v1/model-operations/{operation_id}")

    def cancel_model_operation(self, operation_id):
        return self._request("POST", f"/v1/model-operations/{operation_id}/cancel", {})

    def submit_segment(self, payload):
        return str(self._request("POST", "/v1/jobs/segment", payload)["job_id"])

    def submit_video(self, payload):
        return str(self._request("POST", "/v1/jobs/segment-video", payload)["job_id"])

    def submit_refine(self, payload):
        return str(self._request("POST", "/v1/jobs/refine", payload)["job_id"])

    def submit_inpaint(self, payload):
        return str(self._request("POST", "/v1/jobs/inpaint", payload)["job_id"])

    def preview_trimap(self, payload):
        return self._request("POST", "/v1/preview/trimap", payload, timeout_seconds=60.0)

    def preview_mask_postprocess(self, payload):
        return self._request(
            "POST",
            "/v1/preview/mask-postprocess",
            payload,
            timeout_seconds=60.0,
        )

    def job(self, job_id):
        return self._request("GET", f"/v1/jobs/{job_id}")

    def cancel(self, job_id):
        return self._request("POST", f"/v1/jobs/{job_id}/cancel", {})

    def unload_all(self):
        return self._request("POST", "/v1/providers/unload-all", {})

    def shutdown_server(self):
        return self._request("POST", "/v1/server/shutdown", {})

    def wait(self, job_id, poll_seconds=0.2, timeout_seconds=600.0):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            job = self.job(job_id)
            if job["status"] in ("succeeded", "failed", "cancelled"):
                return job
            time.sleep(poll_seconds)
        raise NukeKyvenClientError(f"Timed out waiting for Kyven job {job_id}.")
