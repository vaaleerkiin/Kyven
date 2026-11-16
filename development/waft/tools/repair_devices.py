"""Re-export existing frozen WAFT weights with input-device tensor placement."""
import io
from pathlib import Path
import zipfile

import torch

from device_portability import make_devices_dynamic

root = Path(__file__).resolve().parents[1]
package = root.parents[1] / 'plugins/Cattery/WAFT'
for source, target in [('WAFT.cat', 'WAFT_FP16.pt'), ('WAFT_FP32.cat', 'WAFT.pt')]:
    # .cat has a Nuke header surrounding a TorchScript zip. Repack just the zip
    # for torch.jit.load; CatFileCreator adds fresh native metadata afterwards.
    archive = io.BytesIO()
    with zipfile.ZipFile(package / source) as original, zipfile.ZipFile(archive, 'w') as output:
        for name in original.namelist():
            output.writestr(name, original.read(name))
    archive.seek(0)
    model = torch.jit.load(archive, map_location='cpu')
    make_devices_dynamic(model)
    destination = root / 'build' / target
    model.save(str(destination))
    print('DEVICE_REPAIR', source, destination, flush=True)
