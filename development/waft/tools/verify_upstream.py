"""Compare the export directly to the original upstream forward methods."""
import json
from pathlib import Path
import types
import cv2
import torch
from export_waft import make_model, PairCore, ROOT

torch.set_num_threads(6)
model = make_model(ROOT / 'checkpoints/waft-a1.pth').cuda().eval()
refiner = model.refine_net
refiner.forward = types.MethodType(type(refiner).forward, refiner)
encoder = model.da_feature.depth_anything.pretrained
encoder.prepare_tokens_with_masks = types.MethodType(type(encoder).prepare_tokens_with_masks, encoder)
core = PairCore(model, 5)
export = torch.jit.load(str(ROOT / 'build/WAFT.pt'), map_location='cuda')
results = []
with torch.inference_mode():
    for h, w in [(160, 240), (241, 353)]:
        torch.manual_seed(7)
        x = torch.rand(1, 6, h, w, device='cuda')
        reference = core(x)
        actual = export(x)
        torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)
        results.append({'shape': [h, w], 'max_error': (actual-reference).abs().max().item()})
        print(results[-1], flush=True)
(ROOT / 'build/upstream_validation.json').write_text(json.dumps(results, indent=2))
print('UPSTREAM_EQUIVALENCE_PASSED', flush=True)
