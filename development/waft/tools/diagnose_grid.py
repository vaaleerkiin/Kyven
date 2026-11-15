import json
from pathlib import Path
import cv2
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
video = cv2.VideoCapture(r'C:\Users\vkame\Downloads\PF0006-01_Clip-1_REC709_2K (1).mp4')
images = []
for frame in [77, 78]:
    video.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
    ok, image = video.read()
    assert ok
    image = cv2.resize(image, (1024, 540), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(root / f'build/user_input_{frame}.png'), image)
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
video.release()
x = torch.from_numpy(np.concatenate(images, 2).transpose(2, 0, 1).copy()).unsqueeze(0).cuda().float()/255.0
torch.set_num_threads(6)
results = {}
with torch.inference_mode():
    for name, filename in [('FP16', 'WAFT_FP16.pt'), ('FP32', 'WAFT.pt')]:
        model = torch.jit.load(str(root / 'build' / filename), map_location='cuda')
        output = model(x)[0].cpu().numpy()
        np.save(root / f'build/user_flow_{name}.npy', output)
        # View both weak background variations and the raw saturated vectors.
        view = np.zeros((540,1024,3), dtype=np.float32)
        view[:, :, :2] = output.transpose(1,2,0)
        cv2.imwrite(str(root / f'build/user_flow_{name}.png'), np.clip(view[:,:,::-1]*255,0,255).astype(np.uint8))
        bg = output[:, 40:280, 700:1000]
        phase = np.stack([bg[:, iy::2, ix::2].mean((1,2)) for iy in range(2) for ix in range(2)])
        results[name] = {'phase_2x2': phase.tolist(),
                         'phase_range': np.ptp(phase, axis=0).tolist()}
        print(name, results[name], flush=True)
        del model
        torch.cuda.empty_cache()
    a = np.load(root / 'build/user_flow_FP16.npy')
    b = np.load(root / 'build/user_flow_FP32.npy')
    results['difference'] = {'mean': float(np.abs(a-b).mean()), 'max': float(np.abs(a-b).max())}
(root / 'build/grid_diagnosis.json').write_text(json.dumps(results, indent=2))
print('GRID_DIAGNOSIS_DONE', results, flush=True)
