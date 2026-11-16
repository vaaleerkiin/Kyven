"""Cold starts, retries and cancellation without a real listener or GPU."""
import runpy
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'hosts/nuke'))
from kyven_nuke import runtime
from kyven_nuke.client import NukeKyvenClientError


class Clock:
    now = 0.0
    on_sleep = None

    def monotonic(self):
        return self.now

    def sleep(self, _seconds):
        self.now += 20
        if self.on_sleep:
            callback, self.on_sleep = self.on_sleep, None
            callback()


@pytest.fixture
def launch(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(runtime, 'time', clock)
    monkeypatch.setattr(runtime, '_server_start_lock', threading.Lock())
    monkeypatch.setattr(runtime, '_startup_cancel', None)
    monkeypatch.setattr(runtime, '_stop_generation', 0)
    monkeypatch.setattr(runtime, '_server_state', {'status': 'unknown', 'started_at': None})
    monkeypatch.setattr(runtime, '_unload_legacy_servers', lambda: None)
    monkeypatch.setattr(runtime.config, 'root', lambda: tmp_path)
    monkeypatch.setattr(runtime.config, 'runtime_dir', lambda: tmp_path / 'runtime')
    monkeypatch.setattr(runtime.config, 'models_dir', lambda: tmp_path / 'models')
    monkeypatch.setattr(runtime.config, 'python_executable', lambda: Path(sys.executable))
    process = Mock()
    process.poll.return_value = None
    popen = Mock(return_value=process)
    monkeypatch.setattr(runtime.subprocess, 'Popen', popen)
    current = Mock()
    current.health.side_effect = lambda: {'service': 'kyven', 'api_version': runtime.REQUIRED_API_VERSION}
    monkeypatch.setattr(runtime, 'client', Mock(return_value=current))
    return clock, current, process, popen


def unavailable():
    raise NukeKyvenClientError('connection refused')


def test_cold_start_can_take_longer_than_old_30_second_limit(launch):
    clock, current, process, popen = launch
    current.health.side_effect = lambda: unavailable() if clock.now < 60 else {
        'api_version': runtime.REQUIRED_API_VERSION}
    progress = []
    assert runtime.ensure_server(progress_callback=progress.append) is current
    assert clock.now == 60
    assert any('120s' in message for message in progress)
    assert runtime.server_state()['status'] == 'ready'
    popen.assert_called_once()
    process.terminate.assert_not_called()
    assert runtime._startup_cancel is None


def test_timeout_reaps_child_and_a_new_start_can_retry(launch):
    _, current, process, popen = launch
    current.health.side_effect = unavailable
    with pytest.raises(RuntimeError, match='did not become ready'):
        runtime.ensure_server(timeout_seconds=1)
    process.terminate.assert_called_once()
    process.wait.assert_called_once()
    assert runtime.server_state()['status'] == 'failed'
    assert runtime._startup_cancel is None
    current.health.side_effect = lambda: {'api_version': runtime.REQUIRED_API_VERSION} if popen.call_count >= 2 else unavailable()
    assert runtime.ensure_server() is current
    assert popen.call_count == 2
    assert runtime.server_state()['status'] == 'ready'


def test_stop_cancels_a_start_before_http_is_available(launch):
    clock, current, process, _ = launch
    current.health.side_effect = unavailable
    def stop():
        assert runtime.stop_server()
    clock.on_sleep = stop
    with pytest.raises(RuntimeError, match='startup cancelled'):
        runtime.ensure_server()
    process.terminate.assert_called_once()
    assert runtime.server_state()['status'] == 'stopped'
    assert runtime._startup_cancel is None


def test_outdated_server_is_restarted_only_under_start_lock(launch, monkeypatch):
    _, current, _, popen = launch
    restarted = []
    def stop():
        assert runtime._server_start_lock.locked()
        restarted.append(True)
    monkeypatch.setattr(runtime, '_stop_running_server', stop)
    current.health.side_effect = lambda: {'api_version': runtime.REQUIRED_API_VERSION if restarted else 1}
    assert runtime.ensure_server() is current
    assert restarted == [True]
    popen.assert_called_once()


def test_parallel_requests_share_one_launch(launch):
    _, current, _, popen = launch
    launched = threading.Event()
    current.health.side_effect = lambda: {'api_version': runtime.REQUIRED_API_VERSION} if launched.is_set() else unavailable()
    def create(*args, **kwargs):
        launched.set()
        return popen.return_value
    popen.side_effect = create
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: runtime.ensure_server(), range(2)))
    assert results == [current, current]
    popen.assert_called_once()


def test_stop_also_cancels_a_start_waiting_for_the_start_lock(launch):
    _, current, _, popen = launch
    current.health.side_effect = unavailable
    waiting = threading.Event()
    runtime._server_start_lock.acquire()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runtime.ensure_server, progress_callback=lambda _: waiting.set())
        try:
            assert waiting.wait(timeout=2)
            runtime.stop_server()
        finally:
            runtime._server_start_lock.release()
        with pytest.raises(RuntimeError, match='startup cancelled'):
            future.result(timeout=2)
    popen.assert_not_called()


def test_launched_wrong_api_fails_immediately_and_cleans_up(launch):
    _, current, process, _ = launch
    current.health.side_effect = lambda: {'api_version': -1}
    with pytest.raises(RuntimeError, match='incompatible'):
        runtime._launch_server(120, None, threading.Event())
    process.terminate.assert_called_once()


def test_entrypoint_prefers_updated_checkout_over_installed_wheel(launch, tmp_path):
    entry = tmp_path / 'src/kyven/server/bootstrap.py'
    entry.parent.mkdir(parents=True)
    entry.touch()
    assert runtime._server_entrypoint(Path(sys.executable)) == [sys.executable, '-I', '-u', str(entry)]
    entry.unlink()
    assert runtime._server_entrypoint(Path(sys.executable))[-2:] == ['-m', 'kyven.server.bootstrap']


def test_checkout_bootstrap_ignores_stale_package_on_sys_path(tmp_path):
    stale = tmp_path / 'kyven'
    stale.mkdir()
    (stale / '__init__.py').write_text('')
    (stale / 'cli.py').write_text('raise RuntimeError("STALE BACKEND WAS IMPORTED")')
    entry = Path(__file__).parents[1] / 'src/kyven/server/bootstrap.py'
    script = (f'import sys,runpy; sys.path.insert(0,{str(tmp_path)!r}); '
              f'sys.argv=[{str(entry)!r},"--help"]; runpy.run_path({str(entry)!r},run_name="__main__")')
    result = subprocess.run([sys.executable, '-I', '-c', script],
                            capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert 'Local AI tools for compositing' in result.stdout


def test_start_and_stop_commands_are_registered_in_nodes_tab_menu(monkeypatch):
    import kyven_nuke

    fake = Mock(spec=['menu', 'pluginAddPath', 'addUpdateUI', 'addOnScriptLoad'])
    monkeypatch.setitem(sys.modules, 'nuke', fake)
    monkeypatch.setattr(kyven_nuke, 'repair_segment_nodes', Mock())
    monkeypatch.setattr(kyven_nuke, 'repair_dlss_nodes', Mock())
    runpy.run_path(str(Path(__file__).parents[1] / 'hosts/nuke/menu.py'))
    fake.menu.assert_called_once_with('Nodes')
    commands = {call.args[0]: call.args[1]
                for call in fake.menu.return_value.addMenu.return_value.addCommand.call_args_list}
    assert commands['Start Kyven Server'] is kyven_nuke.start_server
    assert commands['Stop Kyven Server'] is kyven_nuke.stop_server_from_menu
