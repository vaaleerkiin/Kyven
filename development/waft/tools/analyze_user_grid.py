import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
import json
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
images = {}
metrics = {}
for label in ['raw','clean']:
    image = cv2.imread(str(root / f'build/user_nuke_{label}.exr'), cv2.IMREAD_UNCHANGED)
    assert image is not None
    field = image[:,:,[2,1]]
    assert np.isfinite(field).all()
    images[label] = field
    # Background wall, away from the moving person.
    wall = field[30:300, 760:1000]
    high = wall - cv2.GaussianBlur(wall, (5,5), 1.0)
    metrics[label] = {'background_high_frequency_rms': float(np.sqrt(np.mean(high**2)))}
    preview = np.clip(image[:,:,:3],0,1)**(1/2.2)
    cv2.imwrite(str(root / f'build/user_nuke_{label}.png'), (preview*255).astype(np.uint8))
raw_mask = np.linalg.norm(images['raw'],axis=2) > 1
clean_mask = np.linalg.norm(images['clean'],axis=2) > 1
metrics['silhouette_iou'] = float((raw_mask & clean_mask).sum()/(raw_mask | clean_mask).sum())
metrics['high_frequency_reduction'] = 1 - metrics['clean']['background_high_frequency_rms']/metrics['raw']['background_high_frequency_rms']
assert metrics['high_frequency_reduction'] > 0.2, metrics
assert metrics['silhouette_iou'] > 0.99, metrics
(root / 'build/user_grid_validation.json').write_text(json.dumps(metrics, indent=2))
print('USER_GRID_ANALYSIS_PASSED', metrics)
