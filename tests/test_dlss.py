"""Synthetic DLSS contract tests. No NVIDIA runtime or model downloads are used."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kyven.cache_manifest import manifest_path
from kyven.cancellation import CancellationToken
from kyven.client import KyvenClient, KyvenClientError
from kyven.dlss.images import read_channels, write_exr
from kyven.dlss.models import DlssRequest
from kyven.dlss.provider import ExternalDlssProvider
from kyven.dlss.service import DlssService
from kyven.dlss.temporal import reproject, stabilize
from kyven.errors import ErrorCode, KyvenError
from kyven.models.catalog import ModelCatalog
from kyven.segment.providers.registry import ProviderRegistry
from kyven.segment.service import SegmentService
from kyven.server.app import KyvenServer, ServerConfig
from kyven.server.jobs import JobManager


class SyntheticDlss:
    generation = 0

    def __init__(self) -> None:
        self.calls = []
        self.delta = 0.2
        self.unloaded = False

    def predict(self, request, rgba, motion, mask, depth, reset, cancellation):
        cancellation.raise_if_cancelled()
        self.calls.append({"rgba": rgba.copy(), "motion": motion, "mask": mask,
                           "depth": depth, "reset": reset})
        result = rgba.copy()
        result[..., :3] += self.delta if mask is None else self.delta * mask[..., None]
        return result

    def unload(self):
        self.unloaded = True
        self.generation += 1


@pytest.fixture
def shot(tmp_path):
    pytest.importorskip("OpenEXR")
    source = np.full((8, 12, 4), 0.2, dtype=np.float32)
    source[..., 3] = np.linspace(0, 1, 12)
    write_exr(tmp_path / "source.exr", source)
    vectors = np.zeros_like(source)
    write_exr(tmp_path / "motion.exr", vectors)
    (tmp_path / "worker.exe").write_bytes(b"synthetic worker identity; never executed")
    request = DlssRequest(source=tmp_path / "source.exr", output=tmp_path / "result.exr",
                          worker=tmp_path / "worker.exe", motion=tmp_path / "motion.exr")
    provider = SyntheticDlss()
    registry = ProviderRegistry()
    registry.register("dlss-external", lambda: provider)
    return request, source, provider, DlssService(registry)


def test_alpha_is_delivered_to_inference_as_a_guide_without_premultiplying(shot):
    request, original, provider, service = shot
    request = replace(request, mask_source="source_alpha")
    service.run(request)
    call = provider.calls[-1]
    np.testing.assert_array_equal(call["rgba"][..., :3],
                                  original[..., :3].astype(np.float16).astype(np.float32))
    np.testing.assert_array_equal(call["rgba"][..., 3], 1.0)
    np.testing.assert_array_equal(call["mask"], original[..., 3])
    output = read_channels(request.output)
    np.testing.assert_array_equal(output["A"], original[..., 3])
    np.testing.assert_array_equal(output["R"][:, 0], original[:, 0, 0])
    np.testing.assert_allclose(output["R"], original[..., 0] + 0.2 * original[..., 3], atol=1e-7)


def test_mask_is_still_an_inference_input_when_final_protection_is_disabled(shot):
    request, original, provider, service = shot
    service.run(replace(request, mask_source="input", mask=request.source,
                        protect_outside=False))
    np.testing.assert_array_equal(provider.calls[-1]["mask"], original[..., 3])
    np.testing.assert_allclose(read_channels(request.output)["R"],
                               original[..., 0] + 0.2 * original[..., 3], atol=1e-7)


def test_hdr_negative_rgb_and_alpha_precision_are_preserved(shot):
    request, original, provider, service = shot
    original[..., 0] = 7.5
    original[..., 1] = -0.25
    original[..., 3] = 0.1234567
    write_exr(request.source, original)
    provider.delta = 0
    service.run(request)
    channels = read_channels(request.output)
    np.testing.assert_array_equal(channels["R"], original[..., 0])
    np.testing.assert_array_equal(channels["G"], original[..., 1])
    assert provider.calls[-1]["rgba"][..., :3].min() >= 0
    assert provider.calls[-1]["rgba"][..., :3].max() <= 1
    np.testing.assert_array_equal(channels["A"], original[..., 3])


@pytest.mark.parametrize("value", [1.1, 8.0, 65504.0, 1e6, -10.0])
def test_identity_inference_preserves_hdr_exr_and_alpha_exactly(shot, value):
    request, source, provider, service = shot
    source[..., :3] = (value, value / 2, -0.125)
    source[..., 3] = 0.1234567
    write_exr(request.source, source)
    provider.delta = 0
    result = service.run(request)
    channels = read_channels(request.output)
    for index, name in enumerate(("R", "G", "B", "A")):
        np.testing.assert_array_equal(channels[name], source[..., index])
    assert result["metadata"]["hdr_rolloff"] == "anchored_peak_shoulder_v1"


def test_rolloff_settings_invalidate_cache_and_temporal_history(shot):
    request, _, provider, service = shot
    request = replace(request, temporal=True)
    first = service.run(request)
    second = service.run(replace(request, frame=2, rolloff_knee=0.7))
    assert first["cache_key"] != second["cache_key"]
    assert provider.calls[-1]["reset"]
    assert second["metadata"]["history_reason"] == "settings_or_session_changed"


def test_disabling_rolloff_rejects_half_float_overflow_instead_of_sending_inf(shot):
    request, source, provider, service = shot
    source[..., :3] = 1e6
    write_exr(request.source, source)
    with pytest.raises(KyvenError, match="float16"):
        service.run(replace(request, hdr_rolloff=False))
    assert not provider.calls


@pytest.mark.parametrize("units,x,y", [("pixels", 2, -3), ("uv", 24, -24)])
def test_external_vectors_have_explicit_units_and_y_orientation(shot, units, x, y):
    request, original, provider, service = shot
    vectors = np.ones_like(original)
    write_exr(request.motion, vectors)
    service.run(replace(request, motion_units=units, motion_scale_x=2, motion_scale_y=3))
    motion = provider.calls[-1]["motion"]
    np.testing.assert_allclose(motion[..., 0], x)
    np.testing.assert_allclose(motion[..., 1], y)


def test_low_memory_reduction_scales_vectors_and_restores_full_source(shot):
    request, _, provider, service = shot
    source = np.full((128, 256, 4), 0.5, dtype=np.float32)
    vectors = np.ones_like(source) * 8
    write_exr(request.source, source)
    write_exr(request.motion, vectors)
    result = service.run(replace(request, max_dimension=64))
    assert provider.calls[-1]["rgba"].shape == (32, 64, 4)
    np.testing.assert_allclose(provider.calls[-1]["motion"][..., 0], 2)
    np.testing.assert_allclose(provider.calls[-1]["motion"][..., 1], -2)
    assert read_channels(request.output)["R"].shape == (128, 256)
    assert result["metadata"]["processing_size"] == (64, 32)


def test_temporal_stability_reduces_flicker_and_replays_cached_frames(shot):
    request, _, provider, service = shot
    request = replace(request, temporal=True, temporal_strength=0.75, frame=1001)
    first = service.run(request)
    provider.delta = -0.2
    second = service.run(replace(request, frame=1002))
    assert [c["reset"] for c in provider.calls] == [True, False]
    np.testing.assert_allclose(read_channels(request.output)["R"], 0.3, atol=1e-7)
    assert not second["metadata"]["history_reset"]
    assert first["cache_key"] != second["cache_key"]
    # Repeating a timeline frame resets, rather than reusing an advanced private state.
    third = service.run(replace(request, frame=1002))
    assert len(provider.calls) == 3
    assert third["metadata"]["history_reason"] == "nonconsecutive_frame"


@pytest.mark.parametrize("change,reason", [
    ({"frame": 3}, "nonconsecutive_frame"),
    ({"frame": 0}, "nonconsecutive_frame"),
    ({"frame": 2, "reset": True}, "artist_reset"),
    ({"frame": 2, "session": "another"}, "settings_or_session_changed"),
    ({"frame": 2, "intensity": 0.8}, "settings_or_session_changed"),
    ({"frame": 2, "mask_source": "source_alpha"}, "settings_or_session_changed"),
])
def test_temporal_history_resets_on_discontinuities(shot, change, reason):
    request, _, provider, service = shot
    request = replace(request, temporal=True)
    service.run(request)
    result = service.run(replace(request, **change))
    assert result["metadata"]["history_reason"] == reason
    assert provider.calls[-1]["reset"]


def test_scene_cuts_reject_previous_neural_history(shot):
    request, original, provider, service = shot
    request = replace(request, temporal=True)
    service.run(request)
    original[..., :3] = 4.0
    write_exr(request.source, original)
    result = service.run(replace(request, frame=2))
    assert result["metadata"]["history_reason"] == "scene_cut"
    assert provider.calls[-1]["reset"]


def test_history_key_depends_on_preceding_frames_and_runtime(shot):
    request, original, _, service = shot
    request = replace(request, temporal=True)
    service.run(request)
    first = service.run(replace(request, frame=2))
    original[..., :3] += 0.01
    write_exr(request.source, original)
    service.run(request)
    # Restore identical current frame; only its predecessor changed.
    original[..., :3] -= 0.01
    write_exr(request.source, original)
    second = service.run(replace(request, frame=2))
    assert first["cache_key"] != second["cache_key"]
    (request.worker.parent / "runtime.dll").write_bytes(b"changed runtime")
    result = service.run(replace(request, frame=3))
    assert result["metadata"]["history_reason"] == "settings_or_session_changed"


def test_still_cache_is_deterministic_validated_and_sensitive_to_inputs(shot):
    request, original, provider, service = shot
    first = service.run(request)
    second = service.run(request)
    assert second["metadata"]["cache_hit"]
    assert first["cache_key"] == second["cache_key"]
    assert len(provider.calls) == 1
    request.output.write_bytes(b"corrupt")
    service.run(request)
    assert len(provider.calls) == 2
    write_exr(request.output, original)
    service.run(request)
    assert len(provider.calls) == 3  # Valid but replaced pixels cannot reuse an old manifest.
    original[..., 0] += 0.01
    write_exr(request.source, original)
    third = service.run(request)
    assert first["cache_key"] != third["cache_key"]
    assert len(provider.calls) == 4
    request.motion.write_bytes(b"invalid motion")
    with pytest.raises(KyvenError):
        service.run(request)


def test_provider_switch_unloads_dlss_and_resets_its_history(shot):
    request, _, provider, service = shot
    request = replace(request, temporal=True)
    service.run(request)
    service.registry.register("other", SyntheticDlss)
    service.registry.activate("other")
    assert provider.unloaded
    result = service.run(replace(request, frame=2))
    assert result["metadata"]["history_reset"]


def test_cancellation_does_not_publish_output_or_manifest(shot):
    request, _, provider, service = shot
    token = CancellationToken()
    original_predict = provider.predict

    def cancelling(*args):
        result = original_predict(*args)
        token.cancel()
        return result

    provider.predict = cancelling
    with pytest.raises(KyvenError) as error:
        service.run(request, token)
    assert error.value.code == ErrorCode.CANCELLED
    assert not request.output.exists()
    assert not manifest_path(request.output).exists()
    assert provider.unloaded


def test_reprojection_moves_history_and_rejects_out_of_bounds():
    previous = np.arange(12, dtype=np.float32).reshape(3, 4)
    motion = np.zeros((3, 4, 2), dtype=np.float32)
    motion[..., 0] = -1
    warped, valid = reproject(previous, motion)
    np.testing.assert_array_equal(warped[:, 1:], previous[:, :-1])
    assert not valid[:, 0].any()
    assert valid[:, 1:].all()


def test_reprojection_rejects_disocclusion_and_changed_masks():
    source = np.full((3, 4, 3), 0.2, dtype=np.float32)
    old = source.copy()
    old[0] = 1
    residual = np.zeros_like(source)
    old_residual = np.ones_like(source)
    mask = np.ones((3, 4), dtype=np.float32)
    old_mask = mask.copy()
    old_mask[1] = 0
    result = stabilize(residual, old_residual, old, source, mask, old_mask,
                       np.zeros((3, 4, 2)), 0.75, 0.1)
    np.testing.assert_array_equal(result[:2], 0)
    np.testing.assert_allclose(result[2], 0.75)


@pytest.mark.parametrize("change", [
    {"temporal": True, "motion": None}, {"mask_source": "input"},
    {"intensity": float("nan")}, {"rejection": 0}, {"timeout": 0},
    {"max_dimension": 0}, {"motion_units": "forward"}, {"style": 8},
    {"rolloff_knee": 0}, {"rolloff_knee": float("nan")}, {"hdr_rolloff": "yes"},
    {"output": Path("relative.exr")},
])
def test_invalid_requests_are_structured_errors(shot, change):
    request, _, _, _ = shot
    with pytest.raises(KyvenError) as error:
        replace(request, **change).validate()
    assert error.value.code == ErrorCode.INVALID_REQUEST


def test_invalid_payload_types_do_not_escape_server_validation(shot):
    request, _, _, _ = shot
    for fields in ({"temporal": "false"}, {"frame": 1.5}, {"intensity": "high"},
                   {"unknown": 1}, {"source": []}):
        with pytest.raises(KyvenError):
            DlssRequest.from_payload(dict(source=str(request.source),
                                         output=str(request.output), worker=str(request.worker),
                                         **{k: v for k, v in fields.items() if k != "source"})
                                     | ({"source": []} if "source" in fields else {}))


def test_capabilities_cpu_failure_and_lazy_registry(tmp_path):
    registry = ModelCatalog.builtin().registry(tmp_path, "cpu")
    assert registry.loaded_ids == ()
    provider = registry.get("dlss-external")
    assert provider.capabilities.supports_control_mask
    assert provider.capabilities.supports_external_motion
    assert provider.capabilities.supports_reduced_resolution
    assert not provider.capabilities.supports_cpu
    assert not provider.capabilities.supports_tiling
    assert provider.capabilities.license_url.startswith("https://")
    request = DlssRequest(tmp_path / "source.exr", tmp_path / "out.exr", tmp_path / "worker.exe")
    with pytest.raises(KyvenError) as error:
        provider.predict(request, np.zeros((2, 2, 4)), None, None, None,
                         True, CancellationToken())
    assert error.value.code == ErrorCode.PROVIDER_UNAVAILABLE
    registry.unload_all()


def test_authenticated_dlss_endpoint_uses_existing_job_queue_and_durable_cache(shot):
    request, _, _, service = shot
    manager = JobManager(SegmentService(service.registry), dlss_service=service)
    server = KyvenServer(ServerConfig("x" * 32, request.source.parent, port=0), manager,
                         service.registry, ModelCatalog.builtin())
    server.start()
    client = KyvenClient(f"http://127.0.0.1:{server.port}", "x" * 32)
    payload = {"source": str(request.source), "output": str(request.output),
               "worker": str(request.worker), "mask_source": "source_alpha"}
    try:
        assert client.health()["api_version"] == 35
        with pytest.raises(KyvenClientError):
            KyvenClient(client.base_url, "y" * 32).submit_dlss(payload)
        with pytest.raises(KyvenClientError):
            client.submit_dlss(dict(payload, temporal="false"))
        job = client.wait(client.submit_dlss(payload), poll_seconds=0.02)
        assert job["status"] == "succeeded", job
        assert Path(job["result"]["output"]).is_file()
        assert manifest_path(request.output).is_file()
    finally:
        server.shutdown()


_FAKE_WORKER = r'''
import json, struct, sys, time
mode, log = sys.argv[1:3]
inp, out = sys.stdin.buffer, sys.stdout.buffer
def read(n):
    data = bytearray()
    while len(data) < n:
        part = inp.read(n - len(data))
        if not part:
            raise SystemExit(0)
        data.extend(part)
    return bytes(data)
h = struct.unpack('<14I4f5If', read(96))
w, height = h[1:3]
if mode == 'hang_setup':
    time.sleep(30)
out.write(struct.pack('<12I', 0x34505553, int(mode != 'bad_setup'), 0,
                      w, height, w, height, 0, 0, 0, 0, 0))
out.flush()
while True:
    f = struct.unpack('<4Iq', read(24))
    pixels = read(w * height * 8)
    motion = read(w * height * 4) if h[18] == 1 else b''
    depth = read(w * height * 4) if f[3] & 1 else b''
    mask = read(w * height * 4) if f[3] & 2 else b''
    with open(log, 'a') as stream:
        stream.write(json.dumps({'header': h, 'frame': f,
            'mask': list(struct.unpack('<' + str(w * height) + 'f', mask)) if mask else [],
            'motion': list(struct.unpack('<' + str(w * height * 2) + 'e', motion)) if motion else []}) + '\n')
    if mode == 'hang_frame':
        time.sleep(30)
    response = (0x3154554F, f[1], 1, len(pixels) + int(mode == 'bad_size'), 1, f[4])
    out.write(struct.pack('<5Iq', *response))
    if mode == 'truncated':
        out.flush()
        raise SystemExit(0)
    out.write(pixels)
    out.flush()
'''


@pytest.fixture
def transport(tmp_path, monkeypatch):
    import kyven.dlss.provider as module

    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER, encoding="utf-8")
    log = tmp_path / "wire.jsonl"
    mode = ["normal"]
    popen = subprocess.Popen
    processes = []

    def launch(args, **kwargs):
        kwargs.pop("creationflags", None)
        child = popen([sys.executable, str(script), mode[0], str(log)], **kwargs)
        processes.append(child)
        return child

    monkeypatch.setattr(module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(Popen=launch, PIPE=subprocess.PIPE,
        DEVNULL=subprocess.DEVNULL, CREATE_NO_WINDOW=0))
    request = DlssRequest(tmp_path / "s.exr", tmp_path / "o.exr", Path(sys.executable), timeout=1)
    provider = ExternalDlssProvider()
    provider.capabilities = replace(provider.capabilities, supports_control_mask=True)
    monkeypatch.setattr(provider, "_check_runtime", lambda request: None)
    monkeypatch.setattr(provider, "_check_mask_worker", lambda request: None)
    yield provider, request, mode, log, processes
    provider.unload()


def test_real_pipe_transport_sends_float_mask_vectors_and_temporal_reset(transport):
    provider, request, _, log, _ = transport
    rgba = np.full((2, 3, 4), 0.25, dtype=np.float32)
    rgba[1, :, 0] = 4.0
    mask = np.linspace(0, 1, 6, dtype=np.float32).reshape(2, 3)
    motion = np.full((2, 3, 2), -2.5, dtype=np.float32)
    for frame, reset in ((1, True), (2, False)):
        output = provider.predict(replace(request, frame=frame), rgba, motion, mask,
                                  mask, reset, CancellationToken())
        np.testing.assert_array_equal(output, rgba)
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [line["frame"][2] for line in lines] == [1, 0]
    assert lines[0]["header"][18] == 1  # External only; never auto optical flow.
    assert lines[0]["frame"][3] == 3  # Both depth and control mask.
    np.testing.assert_allclose(lines[0]["mask"], mask.ravel())
    np.testing.assert_allclose(lines[0]["motion"], motion.ravel())


@pytest.mark.parametrize("failure", ["bad_setup", "bad_size", "truncated", "hang_setup"])
def test_transport_failure_or_timeout_kills_child_and_returns_structured_error(transport, failure):
    provider, request, mode, _, processes = transport
    mode[0] = failure
    with pytest.raises(KyvenError) as error:
        provider.predict(request, np.ones((2, 3, 4)), None, None, None,
                         True, CancellationToken())
    assert error.value.code == ErrorCode.INFERENCE_FAILED
    assert provider._process is None
    assert all(child.poll() is not None for child in processes)


def test_transport_cancellation_interrupts_blocked_pipe_and_unloads(transport):
    provider, request, mode, log, processes = transport
    mode[0] = "hang_frame"
    token = CancellationToken()

    def cancel_after_read():
        deadline = time.monotonic() + 2
        while not log.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        token.cancel()

    watcher = threading.Thread(target=cancel_after_read)
    watcher.start()
    with pytest.raises(KyvenError) as error:
        provider.predict(request, np.ones((2, 3, 4)), None, None, None, True, token)
    watcher.join()
    assert error.value.code == ErrorCode.CANCELLED
    assert all(child.poll() is not None for child in processes)


def test_explicit_history_reset_recreates_private_native_state(transport):
    provider, request, _, log, processes = transport
    rgba = np.full((2, 3, 4), 0.5, dtype=np.float32)
    for frame in (1, 2):
        provider.predict(replace(request, frame=frame), rgba, None, None, None,
                         True, CancellationToken())
    assert len(processes) == 2
    assert processes[0].poll() is not None
    frames = [json.loads(line)["frame"] for line in log.read_text().splitlines()]
    assert [frame[1] for frame in frames] == [0, 0]
    assert [frame[2] for frame in frames] == [1, 1]


def test_missing_runtime_is_reported_before_launching_worker(shot):
    request, _, _, _ = shot
    with pytest.raises(KyvenError) as error:
        ExternalDlssProvider._check_runtime(request)
    assert error.value.code == ErrorCode.PROVIDER_UNAVAILABLE
    assert "nvngx_dlssnr.dll" in error.value.message


def test_legacy_worker_is_rejected_for_masks_with_model_manager_update_action(shot, monkeypatch):
    request, _, _, _ = shot
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="KyvenDLSSWorker 1.0 D5V5"))
    with pytest.raises(KyvenError) as error:
        ExternalDlssProvider._check_mask_worker(request)
    assert error.value.code == ErrorCode.PROVIDER_UNAVAILABLE
    assert "Model Manager" in error.value.suggested_action
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(stdout="KyvenDLSSWorker 2.0 control-mask=rgba-v1"))
    ExternalDlssProvider._check_mask_worker(request)
