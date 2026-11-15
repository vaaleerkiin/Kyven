"""Package only the runtime files, preserving the WAFT top-level folder."""
import hashlib
import json
from pathlib import Path
import zipfile

root = Path(__file__).resolve().parents[1]
package = root / 'WAFT'
validation = {}
report_origins = {}
for name in ['export_validation', 'upstream_validation', 'fp16_validation', 'nuke_validation', 'resolution_v3_validation', 'benchmark_FP32', 'benchmark_FP16', 'filter_validation', 'grid_diagnosis', 'user_grid_validation']:
    report = root / 'build' / (name + '.json')
    if not report.is_file():
        report = root / 'reports' / (name + '.json')
    validation[name] = json.loads(report.read_text())
    report_origins[name] = 'current build' if report.parent.name == 'build' else 'saved baseline'
validation['report_origins'] = report_origins
validation['source_commit'] = json.loads((root / 'provenance.json').read_text())['source_commit']
model = package / 'Cattery/WAFT/WAFT.cat'
validation['cat_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
validation['precision'] = 'FP16 on GPU; FP32 flow accumulation; optional FP32 model for CPU'
validation['package_version'] = 4
validation['default_processing_long_edge'] = 512
(package / 'validation.json').write_text(json.dumps(validation, indent=2))
output = root / 'WAFT-Cattery-Nuke16.zip'
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
    for file in sorted(package.rglob('*')):
        if file.is_file() and '__pycache__' not in file.parts and file.suffix != '.pyc':
            archive.write(file, file.relative_to(root))
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
print(output, output.stat().st_size, flush=True)
