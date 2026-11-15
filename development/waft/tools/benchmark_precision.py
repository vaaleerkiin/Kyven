"""Compare both exported models in separate fresh CUDA processes."""
import json
from pathlib import Path
import sys
import time
import torch

root = Path(__file__).resolve().parents[1]
variant = sys.argv[1]
torch.set_num_threads(6)
torch.manual_seed(5)
path = root / 'build' / ('WAFT_FP16.pt' if variant == 'FP16' else 'WAFT.pt')
with torch.inference_mode():
    model = torch.jit.load(str(path), map_location='cuda')
    x = torch.rand(1, 6, 288, 512, device='cuda')
    for _ in range(2):
        result = model(x)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(5):
        result = model(x)
    torch.cuda.synchronize()
    report = {'precision': variant, 'working_size': [512, 288],
              'seconds_per_pair': (time.perf_counter()-start)/5,
              'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
              'model_bytes': path.stat().st_size}
(root / f'build/benchmark_{variant}.json').write_text(json.dumps(report, indent=2))
print(report, flush=True)
