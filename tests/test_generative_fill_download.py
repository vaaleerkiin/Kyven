"""Installer downloads recover from expired redirects and interrupted transfers."""
import hashlib
import io
import urllib.error

from kyven.generative_fill import assets


class Response(io.BytesIO):
    def __init__(self, payload, status=200, content_range=""):
        super().__init__(payload)
        self.status = status
        self.headers = {"Content-Range": content_range}


def fixture(monkeypatch):
    data = b"a pinned model fixture"
    monkeypatch.setattr(assets, "manifest", lambda: {"files": [{
        "path": "model.bin", "url": "https://example.invalid/pinned/model.bin",
        "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}]})
    monkeypatch.setattr(assets.time, "sleep", lambda seconds: None)
    return data


def test_expired_redirect_gets_a_fresh_url_and_retries(tmp_path, monkeypatch):
    data = fixture(monkeypatch)
    calls = []

    def download(request, **kwargs):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 403, "Expired", {}, None)
        return Response(data)

    monkeypatch.setattr(assets.urllib.request, "urlopen", download)
    assets.download_bundle(tmp_path)
    assert calls[0] != calls[1]
    assert (tmp_path / "model.bin").read_bytes() == data


def test_partial_model_resumes_at_the_verified_byte_range(tmp_path, monkeypatch):
    data = fixture(monkeypatch)
    (tmp_path / "model.bin.download").write_bytes(data[:4])

    def download(request, **kwargs):
        assert request.get_header("Range") == "bytes=4-"
        return Response(data[4:], 206, f"bytes 4-{len(data)-1}/{len(data)}")

    monkeypatch.setattr(assets.urllib.request, "urlopen", download)
    assets.download_bundle(tmp_path)
    assert (tmp_path / "model.bin").read_bytes() == data


def test_bad_checksum_retries_without_publishing_corrupt_weights(tmp_path, monkeypatch):
    data = fixture(monkeypatch)
    responses = iter([Response(b"corrupt"), Response(data)])
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **kw: next(responses))
    assets.download_bundle(tmp_path)
    assert (tmp_path / "model.bin").read_bytes() == data
