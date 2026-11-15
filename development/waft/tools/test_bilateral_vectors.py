import json
from pathlib import Path
import statistics
import nuke

root = Path(__file__).resolve().parents[1]
source = nuke.nodes.Constant()
source['format'].setValue(nuke.addFormat('128 64 1 VectorFilterTest'))
field = nuke.nodes.Expression(inputs=[source])
field['expr0'].setValue('(x<64?3:8)+0.05*(2*(floor(x)%2)-1)*(2*(floor(y)%2)-1)')
field['expr1'].setValue('1.25+0.05*(2*(floor(x)%2)-1)*(2*(floor(y)%2)-1)')
filtered = nuke.nodes.Bilateral(inputs=[field])
filtered['Size'].setValue(2)
filtered['PositionalSigma'].setValue(0.7)
filtered['ColorSigma'].setValue(0.25)
raw = [field.sample('red', x+0.5,y+0.5) for y in range(20,28) for x in range(20,28)]
clean = [filtered.sample('red', x+0.5,y+0.5) for y in range(20,28) for x in range(20,28)]
samples = {str(x): filtered.sample('red',x+0.5,32.5) for x in [61,62,63,64,65,66]}
report = {'raw_std': statistics.pstdev(raw), 'filtered_std': statistics.pstdev(clean), 'edge_samples': samples}
assert report['filtered_std'] < report['raw_std'] * 0.4, report
for x in [61,62,63]:
    assert abs(samples[str(x)] - 3) < 0.1, report
for x in [64,65,66]:
    assert abs(samples[str(x)] - 8) < 0.1, report
(root / 'build/filter_validation.json').write_text(json.dumps(report, indent=2))
print('VECTOR_FILTER_PASSED', report, flush=True)
