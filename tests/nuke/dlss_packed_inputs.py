"""Licensed Nuke export/reopen check; fixtures and assertions live in pytest."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'hosts/nuke'), str(ROOT / 'src')]
import kyven_nuke  # noqa: F401 - serialized callbacks need this name.
import nuke
from kyven_nuke import dlss_node, input_cache
from kyven_nuke.color_management import set_data_io

folder = Path(os.environ['KYVEN_DLSS_PACKED_TEST'])
dlss_node._schedule_live = lambda *args, **kwargs: None
dlss_node.resolve_worker = lambda *args: Path(sys.executable)
nuke.root()['format'].setValue(nuke.addFormat('3840 2160 1 PackedProject'))
inputs = []
for label in ('source', 'motion', 'depth', 'mask'):
    read = nuke.nodes.Read(file=(folder / f'{label}.exr').as_posix())
    set_data_io(read)
    inputs.append(read)
node = dlss_node.create_dlss_node()
node['live_mode'].setValue(False)
node['cache_root'].setValue((folder / 'cache').as_posix())
node['mask_source'].setValue('Mask Input')
node['temporal'].setValue(True)
node['motion_consistency'].setValue(True)
for index, read in enumerate(inputs):
    node.setInput(index, read)
writer, path = input_cache._specs(node, 1, 1, 1)[0]
start = time.perf_counter()
assert input_cache.export_input(nuke, node, writer, 1, path)
seconds = time.perf_counter() - start
timestamp = path.stat().st_mtime_ns
assert not input_cache.export_input(nuke, node, writer, 1, path)
assert path.stat().st_mtime_ns == timestamp
assert input_cache.ready(node, path, 1)
assert len(list(path.parent.glob('*.exr'))) == 1
# Compare to the previous four-Write, float, ZIP1 transport on the same inputs.
baseline = []
start = time.perf_counter()
for index, read in enumerate(inputs):
    out = folder / f'baseline-{index}.exr'
    old = nuke.nodes.Write(file=out.as_posix(), file_type='exr')
    old.setInput(0, read)
    old['channels'].setValue('rgba' if index == 0 else 'all')
    old['datatype'].setValue('32 bit float')
    old['autocrop'].setValue(False)
    set_data_io(old)
    nuke.execute(old, 1, 1)
    baseline.append(out.stat().st_size)
    nuke.delete(old)
baseline_seconds = time.perf_counter() - start
identity, name = node['kyven_uuid'].value(), node.name()
script = folder / 'packed.nk'
nuke.scriptSaveAs(str(script), overwrite=1)
nuke.scriptClear()
nuke.scriptOpen(str(script))
dlss_node.repair_dlss_nodes()
node = nuke.toNode(name)
assert node['kyven_uuid'].value() == identity
assert input_cache.ready(node, path, 1)
assert len([child for child in node.nodes() if child.Class() == 'Write']) == 1
report = {'packed': str(path), 'packed_bytes': path.stat().st_size,
          'separate_bytes': sum(baseline), 'packed_seconds': seconds,
          'separate_seconds': baseline_seconds, 'reopened_uuid': identity}
(folder / 'report.json').write_text(json.dumps(report, indent=2))
print('DLSS_PACKED_NUKE_PASSED', report, flush=True)
