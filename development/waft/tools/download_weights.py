"""Download the WAFT-a1 checkpoint recommended by the upstream authors."""
from pathlib import Path
import html
import hashlib
import json
import re
import urllib.parse
import urllib.request

root = Path(__file__).resolve().parents[1]
target = root / 'checkpoints' / 'waft-a1.pth'
target.parent.mkdir(parents=True, exist_ok=True)
expected_sha256 = json.loads((root / 'provenance.json').read_text())['checkpoint']['sha256']
if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected_sha256:
    print('Checkpoint already verified:', target)
    raise SystemExit(0)
file_id = '1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC'
url = 'https://drive.google.com/uc?export=download&id=' + file_id
with urllib.request.urlopen(url) as response:
    page = response.read().decode()
action = html.unescape(re.search(r'<form[^>]+action="([^"]+)"', page)[1])
params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page))
url = action + '?' + urllib.parse.urlencode(params)
print('Downloading upstream checkpoint to', target, flush=True)
temporary = target.with_suffix('.part')
with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as out:
    if 'text/html' in response.headers.get('Content-Type', ''):
        raise RuntimeError('Google Drive returned HTML instead of weights')
    total = 0
    while block := response.read(8 * 1024 * 1024):
        out.write(block)
        total += len(block)
        print(round(total / 1024**2), 'MiB', flush=True)
if hashlib.sha256(temporary.read_bytes()).hexdigest() != expected_sha256:
    raise RuntimeError('Checkpoint SHA-256 mismatch; destination was not replaced')
temporary.replace(target)
print('Done', target.stat().st_size, flush=True)
