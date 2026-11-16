"""Compare repaired weights after CPU load/GPU transfer against original output."""
import io
import json
from pathlib import Path
import zipfile

import torch
import torch.nn.functional as F

root = Path(__file__).resolve().parents[1]
package = root.parents[1] / 'plugins/Cattery/WAFT'
torch.set_num_threads(6)
torch.manual_seed(42)
assert torch.cuda.is_available(), 'CUDA is required for the regression'
report = []
for source, repaired in [('WAFT.cat', 'WAFT_FP16.pt'), ('WAFT_FP32.cat', 'WAFT.pt')]:
    archive = io.BytesIO()
    with zipfile.ZipFile(package / source) as original, zipfile.ZipFile(archive, 'w') as output:
        for name in original.namelist():
            output.writestr(name, original.read(name))
    archive.seek(0)
    reference = torch.jit.load(archive, map_location='cuda')
    if source == 'WAFT.cat':
        archive.seek(0)
        old = torch.jit.load(archive, map_location='cpu').cuda()
        try:
            old(torch.rand(1, 6, 128, 192, device='cuda'))
        except RuntimeError as error:
            assert 'same device' in str(error), str(error)[-1000:]
            print('ORIGINAL_CPU_LOAD_GPU_TRANSFER_REPRODUCED', flush=True)
        else:
            print('SOURCE_ALREADY_PORTABLE', flush=True)
        del old
    # Unlike map_location='cuda', this leaves frozen constants on CPU. This
    # models a host that loads the file first and moves the module afterwards.
    model = torch.jit.load(str(root / 'build' / repaired), map_location='cpu').cuda()
    with torch.inference_mode():
        for height, width in [(128, 192), (160, 240), (288, 512)]:
            image = F.avg_pool2d(torch.rand(1, 3, height, width, device='cuda'),
                                 5, stride=1, padding=2)
            target = torch.roll(image, shifts=(3, 4), dims=(2, 3))
            x = torch.cat((image, target), dim=1)
            expected = reference(x)
            actual = model(x)
            tolerance = 0.02 if source == 'WAFT.cat' else 0.002
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            assert torch.isfinite(actual).all()
            case = {'model': source, 'size': [height, width],
                    'max_error': (actual - expected).abs().max().item()}
            if source == 'WAFT_FP32.cat' and height == 128:
                cpu = model.cpu()(x.cpu())
                torch.testing.assert_close(cpu, expected.cpu(), rtol=0.02, atol=0.02)
                case['cpu_return_max_error'] = (cpu - expected.cpu()).abs().max().item()
                model.cuda()
            report.append(case)
            print('PORTABILITY_CASE', case, flush=True)
    del reference, model
    torch.cuda.empty_cache()
(root / 'build/portability_validation.json').write_text(json.dumps(report, indent=2))
print('WAFT_FROZEN_TENSOR_PORTABILITY_PASSED', flush=True)
