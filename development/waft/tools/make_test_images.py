from pathlib import Path
import cv2
import numpy as np

root = Path(__file__).resolve().parents[1]
(root / 'build').mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(12)
image = rng.integers(0, 256, (160, 240, 3), dtype=np.uint8)
image = cv2.GaussianBlur(image, (5, 5), 0.8)
cv2.rectangle(image, (40, 30), (100, 90), (50, 170, 230), -1)
cv2.circle(image, (160, 100), 24, (180, 50, 80), -1)
cv2.putText(image, 'WAFT', (55, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
for frame in range(1, 4):
    transform = np.float32([[1, 0, 4*(frame-2)], [0, 1, 3*(frame-2)]])
    result = cv2.warpAffine(image, transform, (240, 160), borderMode=cv2.BORDER_REFLECT)
    cv2.imwrite(str(root / f'build/test_input.{frame:04d}.png'), result)
