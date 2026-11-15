"""Reproduce the user's frame 77 with raw and cleaned motion fields."""
from pathlib import Path
import time
import nuke

root = Path(__file__).resolve().parents[1]
nuke.pluginAddPath((root / 'WAFT/Cattery/WAFT').as_posix())
nuke.root()['format'].setValue(nuke.addFormat('2048 1556 1 UserProject'))
nuke.root()['last_frame'].setValue(212)
nuke.frame(77)
read = nuke.nodes.Read(file=r'C:/Users/vkame/Downloads/PF0006-01_Clip-1_REC709_2K (1).mp4', first=1, last=212)
source = nuke.nodes.Reformat(inputs=[read])
source['type'].setValue('scale')
source['scale'].setValue(0.5)
node = nuke.nodes.WAFT(inputs=[source])
node['processing'].setValue(3)
node['output_mode'].setValue(0)
display = nuke.nodes.Expression(inputs=[node])
display['expr0'].setValue('backward.u')
display['expr1'].setValue('backward.v')
display['expr2'].setValue('0')
display['expr3'].setValue('1')
write = nuke.nodes.Write(inputs=[display])
write['file_type'].setValue('exr')
write['channels'].setValue('rgba')
write['datatype'].setValue('32 bit float')
for strength, label in [(0, 'raw'), (1, 'clean')]:
    node['grid_cleanup'].setValue(strength)
    write['file'].setValue((root / f'build/user_nuke_{label}.exr').as_posix())
    start = time.perf_counter()
    nuke.execute(write, 77, 77)
    b = node.bbox()
    assert [b.x(), b.y(), b.w(), b.h()] == [0, 0, 1024, 540]
    print('USER_RENDER', label, time.perf_counter()-start, flush=True)
nuke.scriptSave((root / 'build/user_grid_v4_review.nk').as_posix())
print('USER_GRID_RENDER_PASSED', flush=True)
