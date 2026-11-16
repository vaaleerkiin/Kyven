"""Real Nuke cold start/stop and offline cache playback."""
import json
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'hosts/nuke'), str(ROOT / 'src')]
import kyven_nuke  # noqa: F401 - Nuke callbacks use this name.
import nuke
from kyven_nuke import dlss_node, runtime

folder = Path(os.environ['KYVEN_SERVER_CONTROL_TEST'])
dlss_node._schedule_live = lambda *args, **kwargs: None
source = nuke.nodes.Constant(format=nuke.addFormat('2 2 1 CacheControls').name())
source['color'].setValue([0.7, 0.7, 0.7, 0.33])
node = dlss_node.create_dlss_node()
node.setInput(0, source)
node['cache_root'].setValue((folder / 'cache').as_posix())
path = Path(str(dlss_node._public_pattern(node)) % 1)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes((folder / 'cached.exr').read_bytes())
node['output_mode'].setValue('Source')
node['live_mode'].setValue(True)
dlss_node.load_cache(node)
assert node['output_mode'].value() == 'Result'
assert not node['live_mode'].value()
assert abs(node.sample('rgba.red', 0.5, 0.5) - 0.2) < 1e-6
assert abs(node.sample('rgba.alpha', 0.5, 0.5) - 0.33) < 1e-6
node.setInput(0, None)
dlss_node.load_cache(node)
assert abs(node.sample('rgba.red', 0.5, 0.5) - 0.2) < 1e-6
assert abs(node.sample('rgba.alpha', 0.5, 0.5) - 0.75) < 1e-6
node['kyven_busy'].setValue(True)
script = folder / 'busy.nk'
name = node.name()
nuke.scriptSaveAs(str(script), overwrite=1)
nuke.scriptClear()
nuke.scriptOpen(str(script))
dlss_node.repair_dlss_nodes()
node = nuke.toNode(name)
assert not node['kyven_busy'].value()
with socket.socket() as probe:
    probe.bind(('127.0.0.1', 0))
    runtime.PORT = probe.getsockname()[1]
runtime.LEGACY_PORTS = ()
runtime.config.runtime_dir = lambda: folder / 'runtime'
runtime.config.models_dir = lambda: folder / 'models'
progress = []
try:
    client = runtime.ensure_server(progress_callback=progress.append)
    assert client.health()['api_version'] == runtime.REQUIRED_API_VERSION
    assert runtime.ensure_server().health()['service'] == 'kyven'
finally:
    runtime.stop_server()
assert runtime.server_state()['status'] == 'stopped'
(folder / 'controls.json').write_text(json.dumps({'passed': True, 'progress': progress}, indent=2))
print('DLSS_SERVER_AND_CACHE_CONTROLS_PASSED', flush=True)
